#!/usr/bin/env python3
"""P15: TUM RGB-D 上不依赖 MoGe 的 latent warp 可行性实验。"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_geometry_alignment import align_depth_to_latent
from latent_geometry_head import points_from_depth
from latent_reprojection_loss import compare_warp_to_copy, forward_splat_latent
from vae import WanVAE

FR1_K = np.array(
    [[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-bin", type=int, default=4)
    parser.add_argument("--candidate-stride", type=int, default=4)
    parser.add_argument("--frame-deltas", type=int, nargs="+", default=(2, 4, 8, 16, 32))
    parser.add_argument("--max-time-offset", type=float, default=0.03)
    parser.add_argument("--visual-count", type=int, default=6)
    return parser.parse_args()


def timed_rows(path: Path, value_count: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != value_count + 1:
            raise ValueError(f"unexpected row in {path}: {line}")
        rows.append((float(fields[0]), tuple(fields[1:])))
    if not rows:
        raise RuntimeError(f"no records in {path}")
    return rows


def closest(rows, timestamp: float):
    times = [row[0] for row in rows]
    index = bisect.bisect_left(times, timestamp)
    choices = [max(0, index - 1), min(len(rows) - 1, index)]
    return min((rows[item] for item in choices), key=lambda row: abs(row[0] - timestamp))


def camera_to_world(raw_values) -> np.ndarray:
    tx, ty, tz, qx, qy, qz, qw = map(float, raw_values)
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = (tx, ty, tz)
    return pose


def synchronize(root: Path, tolerance: float):
    rgb = timed_rows(root / "rgb.txt", 1)
    depth = timed_rows(root / "depth.txt", 1)
    poses = timed_rows(root / "groundtruth.txt", 7)
    records = []
    for rgb_time, (rgb_path,) in rgb:
        depth_row = closest(depth, rgb_time)
        pose_row = closest(poses, rgb_time)
        if abs(depth_row[0] - rgb_time) <= tolerance and abs(pose_row[0] - rgb_time) <= tolerance:
            records.append(
                {
                    "timestamp": rgb_time,
                    "rgb": root / rgb_path,
                    "depth": root / depth_row[1][0],
                    "depth_offset_s": depth_row[0] - rgb_time,
                    "pose_offset_s": pose_row[0] - rgb_time,
                    "camera_to_world": camera_to_world(pose_row[1]),
                }
            )
    if len(records) < 2:
        raise RuntimeError("insufficient synchronized RGB-depth-pose records")
    return records


def crop_box(height: int, width: int):
    if height / width > 352 / 640:
        crop_height = int(width * 352 / 640)
        return 0, (height - crop_height) // 2, width, crop_height
    crop_width = int(height * 640 / 352)
    return (width - crop_width) // 2, 0, crop_width, height


def rgb_tensor(path: Path):
    image = Image.open(path).convert("RGB")
    crop = crop_box(image.height, image.width)
    left, top, width, height = crop
    image = image.crop((left, top, left + width, top + height))
    image = image.resize((640, 352), Image.Resampling.BICUBIC)
    value = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(value).permute(2, 0, 1)[None, :, None], crop


def depth_tensor(path: Path, crop):
    raw = np.asarray(Image.open(path), dtype=np.uint16)
    left, top, width, height = crop
    depth = torch.from_numpy(raw.astype(np.float32) / 5000.0)[None, None]
    depth = depth[..., top : top + height, left : left + width]
    depth = F.interpolate(depth, size=(352, 640), mode="nearest-exact")
    valid = ((depth > 0.2) & (depth < 8.0)).float()
    depth = torch.where(valid.bool(), depth, torch.zeros_like(depth))
    affine = np.array(
        [[640 / width, 0.0, -left * 640 / width],
         [0.0, 352 / height, -top * 352 / height],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    normalized_k = affine @ FR1_K
    normalized_k[0] /= 640
    normalized_k[1] /= 352
    return depth, valid, torch.from_numpy(normalized_k)[None]


def rotation_radians(transform: np.ndarray) -> float:
    cosine = np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(math.acos(cosine))


def select_pairs(records, args):
    bins = {"subcell": [], "moderate_1to4": [], "hard_ge4": []}
    for delta in args.frame_deltas:
        for source_index in range(0, len(records) - delta, args.candidate_stride):
            target_index = source_index + delta
            transform = (
                np.linalg.inv(records[target_index]["camera_to_world"])
                @ records[source_index]["camera_to_world"]
            )
            raw = np.asarray(Image.open(records[source_index]["depth"]), dtype=np.uint16)
            measured = raw[(raw > 1000) & (raw < 40000)].astype(np.float32) / 5000.0
            if measured.size == 0:
                continue
            translation = float(np.linalg.norm(transform[:3, 3]))
            angle = rotation_radians(transform)
            motion = (FR1_K[0, 0] / 8) * (translation / float(np.median(measured)) + angle)
            key = "subcell" if motion < 1 else "moderate_1to4" if motion < 4 else "hard_ge4"
            bins[key].append(
                {
                    "source_index": source_index,
                    "target_index": target_index,
                    "delta": delta,
                    "translation_m": translation,
                    "rotation_deg": math.degrees(angle),
                    "expected_motion_cells": motion,
                    "source_to_target": transform,
                    "motion_bin": key,
                }
            )
    selected = []
    for values in bins.values():
        values.sort(key=lambda item: item["expected_motion_cells"])
        indices = np.linspace(0, len(values) - 1, min(len(values), args.pairs_per_bin), dtype=int)
        selected.extend(values[index] for index in indices)
    if not selected:
        raise RuntimeError("no valid motion pairs")
    return selected, {key: len(value) for key, value in bins.items()}


def psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = (first.float() - second.float()).square().mean().clamp_min(1e-12)
    return float((10 * torch.log10(torch.tensor(4.0, device=mse.device) / mse)).cpu())


def global_ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    mean_first, mean_second = first.mean(), second.mean()
    var_first, var_second = first.var(unbiased=False), second.var(unbiased=False)
    covariance = ((first - mean_first) * (second - mean_second)).mean()
    value = (
        (2 * mean_first * mean_second + 0.02**2)
        * (2 * covariance + 0.06**2)
        / (
            (mean_first.square() + mean_second.square() + 0.02**2)
            * (var_first + var_second + 0.06**2)
        )
    )
    return float(value.cpu())


def image_from_tensor(value: torch.Tensor) -> Image.Image:
    array = (value[0, :, 0].detach().cpu().clamp(-1, 1) + 1) * 127.5
    return Image.fromarray(array.permute(1, 2, 0).byte().numpy())


def montage(path: Path, target, copy, composite):
    canvas = Image.new("RGB", (1920, 352))
    for index, value in enumerate((target, copy, composite)):
        canvas.paste(image_from_tensor(value), (640 * index, 0))
    canvas.save(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def aggregate(rows, field):
    result = {}
    for group in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == group]
        mean = lambda name: float(np.mean([row[name] for row in subset]))
        result[group] = {
            "rows": len(subset),
            "warp_latent_l1": mean("warp_latent_l1"),
            "copy_latent_l1_same_support": mean("copy_latent_l1"),
            "warp_l1_win_rate": float(np.mean([row["warp_latent_l1"] < row["copy_latent_l1"] for row in subset])),
            "warp_cosine": mean("warp_cosine"),
            "copy_cosine_same_support": mean("copy_cosine"),
            "coverage": mean("coverage"),
            "hole_ratio": mean("hole_ratio"),
            "decoded_composite_psnr": mean("decoded_composite_psnr"),
            "decoded_copy_psnr": mean("decoded_copy_psnr"),
            "decoded_composite_ssim": mean("decoded_composite_ssim"),
            "decoded_copy_ssim": mean("decoded_copy_ssim"),
        }
    return result


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    records = synchronize(args.sequence_root, args.max_time_offset)
    pairs, candidate_bins = select_pairs(records, args)
    args.output.mkdir(parents=True)
    visuals = args.output / "visuals"
    visuals.mkdir()

    device = torch.device("cuda:0")
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint))
    vae = vae.to(device, torch.bfloat16).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    methods = ("center", "average", "median", "minimum")
    rows = []
    visual_index = 0
    for pair_index, pair in enumerate(pairs):
        source, target = records[pair["source_index"]], records[pair["target_index"]]
        source_rgb, crop = rgb_tensor(source["rgb"])
        target_rgb, target_crop = rgb_tensor(target["rgb"])
        if crop != target_crop:
            raise RuntimeError("source and target RGB transforms differ")
        depth, valid, intrinsics = depth_tensor(source["depth"], crop)
        with torch.inference_mode():
            source_latent = vae.encode(
                source_rgb.to(device, torch.bfloat16), device=device, **tiler
            )[:, :, 0].float().to(device)
            target_latent = vae.encode(
                target_rgb.to(device, torch.bfloat16), device=device, **tiler
            )[:, :, 0].float().to(device)
            decoded_copy = vae.decode(
                source_latent.to(torch.bfloat16).unsqueeze(2), device=device, **tiler
            ).float().to(device)
        for method in methods:
            aligned_depth, aligned_valid = align_depth_to_latent(depth, valid, (44, 80), method)
            points = points_from_depth(aligned_depth.to(device), intrinsics.to(device))
            transform = torch.from_numpy(pair["source_to_target"]).float().unsqueeze(0).to(device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            warp = forward_splat_latent(
                source_latent, points, aligned_valid.to(device), intrinsics.to(device), transform
            )
            torch.cuda.synchronize()
            warp_ms = (time.perf_counter() - start) * 1000
            comparison = compare_warp_to_copy(warp, source_latent, target_latent)
            with torch.inference_mode():
                decoded_composite = vae.decode(
                    comparison["composite"].to(torch.bfloat16).unsqueeze(2),
                    device=device,
                    **tiler,
                ).float().to(device)
            target_device = target_rgb.to(device)
            rows.append(
                {
                    "pair_index": pair_index,
                    "source_timestamp": source["timestamp"],
                    "target_timestamp": target["timestamp"],
                    "frame_delta": pair["delta"],
                    "motion_bin": pair["motion_bin"],
                    "translation_m": pair["translation_m"],
                    "rotation_deg": pair["rotation_deg"],
                    "expected_motion_cells": pair["expected_motion_cells"],
                    "alignment": method,
                    "warp_latent_l1": float(comparison["warp_valid_l1"]),
                    "copy_latent_l1": float(comparison["copy_valid_l1"]),
                    "warp_cosine": float(comparison["warp_valid_cosine_similarity"]),
                    "copy_cosine": float(comparison["copy_valid_cosine_similarity"]),
                    "composite_full_l1": float(comparison["composite_full_l1"]),
                    "copy_full_l1": float(comparison["copy_full_l1"]),
                    "coverage": float(comparison["coverage"]),
                    "hole_ratio": float(comparison["hole_ratio"]),
                    "decoded_composite_psnr": psnr(decoded_composite, target_device),
                    "decoded_copy_psnr": psnr(decoded_copy, target_device),
                    "decoded_composite_ssim": global_ssim(decoded_composite, target_device),
                    "decoded_copy_ssim": global_ssim(decoded_copy, target_device),
                    "decoded_lpips": None,
                    "warp_latency_ms": warp_ms,
                }
            )
            if method == "median" and visual_index < args.visual_count:
                montage(
                    visuals / f"pair_{pair_index:02d}_{pair['motion_bin']}.png",
                    target_device,
                    decoded_copy,
                    decoded_composite,
                )
                visual_index += 1

    report = {
        "schema_version": 1,
        "stage": "P15 real TUM RGB-D GT-geometry latent-warp feasibility",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim": "MoGe-free feasibility: sensor depth + motion-capture pose + calibrated intrinsics",
        "dataset": {
            "name": "TUM RGB-D freiburg1_xyz",
            "source": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
            "sequence_root": str(args.sequence_root.resolve()),
            "synchronized_records": len(records),
            "candidate_motion_bins": candidate_bins,
            "selected_pairs": len(pairs),
            "depth_scale": 5000.0,
            "intrinsics": FR1_K.tolist(),
        },
        "protocol": {
            "pair_selection": "GT-motion proxy binned before target-latent evaluation",
            "metric_support": "Warp and Copy use the identical projected-valid support",
            "hole_policy": "composite fills holes with source latent; never target latent",
            "rgb_transform": "center crop to 640x352; depth uses identical crop",
            "distortion": "official FR1 pinhole K; residual distortion not modeled",
            "lpips": "not measured in this environment",
        },
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": file_sha256(args.vae_checkpoint),
        "repository": repository_state(),
        "gpu": {"name": torch.cuda.get_device_name(0), "visible_device_count": 1},
        "config": {
            "pairs_per_bin": args.pairs_per_bin,
            "candidate_stride": args.candidate_stride,
            "frame_deltas": args.frame_deltas,
            "max_time_offset": args.max_time_offset,
        },
        "aggregates_by_alignment": aggregate(rows, "alignment"),
        "aggregates_by_motion_bin": aggregate(rows, "motion_bin"),
        "rows": rows,
        "evidence_boundary": "one real indoor sequence; GT feasibility evidence, not generalization",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"pairs": len(pairs), "by_alignment": report["aggregates_by_alignment"]}, indent=2))


if __name__ == "__main__":
    main()
