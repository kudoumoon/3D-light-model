#!/usr/bin/env python3
"""构建 Bonn RGB-D 动态场景的冻结 WanVAE latent、GT 几何与位姿 pair。

该脚本只做数据预处理，不训练或适配任何 M1 参数。所有序列统一标记为 test，
用于严格的跨域动态场景零样本验证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_geometry_alignment import align_depth_to_latent
from tools.run_tum_gt_latent3d_feasibility import crop_box, synchronize
from vae import WanVAE


BONN_K = np.array(
    [[542.8228149414062, 0.0, 315.593505859375],
     [0.0, 542.5768432617188, 237.756103515625],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, action="append", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--pair-deltas", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--max-time-offset", type=float, default=0.03)
    args = parser.parse_args()
    if len(args.sequence_root) != len(args.scene):
        raise ValueError("--sequence-root 与 --scene 数量必须一致")
    if len(set(args.scene)) != len(args.scene):
        raise ValueError("scene 名称必须唯一")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_rgb(path: Path) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    image = Image.open(path).convert("RGB")
    crop = crop_box(image.height, image.width)
    left, top, width, height = crop
    image = image.crop((left, top, left + width, top + height))
    image = image.resize((640, 352), Image.Resampling.BICUBIC)
    value = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(value).permute(2, 0, 1)[None, :, None], crop


def prepare_depth(path: Path, crop: tuple[int, int, int, int]):
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
    intrinsic = affine @ BONN_K
    intrinsic[0] /= 640
    intrinsic[1] /= 352
    latent_depth, _ = align_depth_to_latent(depth, valid, (44, 80), "median")
    latent_valid = (F.avg_pool2d(valid, kernel_size=8, stride=8) >= 0.5).float()
    return latent_depth.float(), latent_valid.float(), torch.from_numpy(intrinsic)[None]


def rotation_angle(rotation: np.ndarray) -> float:
    return float(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))


def build_pairs(scene: str, records: list[dict], sample_ids: list[str], deltas, stride: int):
    rows = []
    for delta in deltas:
        for source_index in range(0, len(records) - delta, stride):
            target_index = source_index + delta
            transform = (
                np.linalg.inv(records[target_index]["camera_to_world"])
                @ records[source_index]["camera_to_world"]
            )
            rvec, _ = cv2.Rodrigues(transform[:3, :3])
            rows.append(
                {
                    "source": sample_ids[source_index],
                    "target": sample_ids[target_index],
                    "scene": f"bonn_{scene}_test",
                    "ok": True,
                    "inliers": 10000,
                    "inlier_ratio": 1.0,
                    "median_reprojection_px": 0.0,
                    "translation_m": float(np.linalg.norm(transform[:3, 3])),
                    "rotation_deg": math.degrees(rotation_angle(transform[:3, :3])),
                    "rvec": rvec.reshape(3).tolist(),
                    "tvec": transform[:3, 3].tolist(),
                    "pose_source": "Bonn RGB-D sensor groundtruth; no image/PnP estimation",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one checked idle GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    for root in args.sequence_root:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)

    device = torch.device("cuda:0")
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(
        device, torch.bfloat16
    ).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    latents, depths, valids, intrinsics = [], [], [], []
    scenes, sample_ids, pair_rows, sequence_stats = [], [], [], []
    started = time.perf_counter()

    for root, scene in zip(args.sequence_root, args.scene):
        synchronized = synchronize(root, args.max_time_offset)
        records = synchronized[:: args.frame_stride]
        local_ids = [f"bonn_{scene}_{index:06d}" for index in range(len(records))]
        for index, (record, sample_id) in enumerate(zip(records, local_ids)):
            rgb, crop = prepare_rgb(record["rgb"])
            depth, valid, intrinsic = prepare_depth(record["depth"], crop)
            with torch.inference_mode():
                latent = vae.encode(
                    rgb.to(device, torch.bfloat16), device=device, **tiler
                )[:, :, 0]
            latents.append(latent.cpu().to(torch.bfloat16))
            depths.append(depth)
            valids.append(valid)
            intrinsics.append(intrinsic.float())
            scenes.append(f"bonn_{scene}_test")
            sample_ids.append(sample_id)
            if (index + 1) % 50 == 0:
                print(json.dumps({"scene": scene, "cached": index + 1, "total": len(records)}), flush=True)
        scene_pairs = build_pairs(scene, records, local_ids, args.pair_deltas, args.pair_stride)
        pair_rows.extend(scene_pairs)
        sequence_stats.append(
            {"scene": scene, "synchronized_frames": len(synchronized),
             "sampled_frames": len(records), "pairs": len(scene_pairs)}
        )

    args.output.mkdir(parents=True)
    cache = {
        "latent": torch.cat(latents),
        "depth": torch.cat(depths),
        "valid": torch.cat(valids),
        "intrinsics": torch.cat(intrinsics),
        "sample_ids": sample_ids,
        "scenes": scenes,
        "sources": [str(root) for root in args.sequence_root],
    }
    torch.save(cache, args.output / "cache.pt")
    (args.output / "pairs.json").write_text(
        json.dumps({"schema_version": 1, "pairs": pair_rows}, indent=2), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    metrics = {
        "schema_version": 1,
        "stage": "Bonn dynamic RGB-D frozen-VAE zero-shot evaluation cache",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "config": {
            "sequence_roots": [str(root) for root in args.sequence_root],
            "scenes": args.scene,
            "frame_stride": args.frame_stride,
            "pair_deltas_on_sampled_frames": args.pair_deltas,
            "pair_stride": args.pair_stride,
            "max_time_offset": args.max_time_offset,
            "rgb_shape": [3, 1, 352, 640],
            "latent_shape": [16, 44, 80],
            "depth_scale": 5000.0,
            "bonn_intrinsics": BONN_K.tolist(),
            "geometry_alignment": "valid-aware median 8x8 pooling",
            "split": "all test; no Bonn training, calibration, or adaptation",
        },
        "sequences": sequence_stats,
        "totals": {"frames": len(sample_ids), "pairs": len(pair_rows)},
        "vae": {"checkpoint": str(args.vae_checkpoint), "sha256": sha256(args.vae_checkpoint), "frozen": True},
        "runtime": {
            "seconds": elapsed,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "evidence_boundary": "Dynamic sequence-level zero-shot benchmark; no dynamic-object segmentation mask.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["totals"], indent=2))
    print("实验已完成")


if __name__ == "__main__":
    main()
