#!/usr/bin/env python3
"""Separate pose/geometry error from VAE-latent alignment error."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reproject_torch import forward_splat_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=25)
    parser.add_argument("--min-inliers", type=int, default=200)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--max-median-reprojection-px", type=float, default=1.5)
    return parser.parse_args()


def repository_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = (first - second).square().mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def global_ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    mean_first, mean_second = first.mean(), second.mean()
    variance_first = first.var(unbiased=False)
    variance_second = second.var(unbiased=False)
    covariance = ((first - mean_first) * (second - mean_second)).mean()
    numerator = (2 * mean_first * mean_second + 0.01**2) * (
        2 * covariance + 0.03**2
    )
    denominator = (
        mean_first.square() + mean_second.square() + 0.01**2
    ) * (variance_first + variance_second + 0.03**2)
    return float(numerator / denominator)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = {record["sample_id"]: record for record in manifest["records"]}
    raw_pairs = json.loads(args.pairs.read_text())["pairs"]
    pairs = [
        pair
        for pair in raw_pairs
        if pair.get("ok")
        and pair.get("inliers", 0) >= args.min_inliers
        and pair.get("inlier_ratio", 0.0) >= args.min_inlier_ratio
        and pair.get("median_reprojection_px", float("inf"))
        <= args.max_median_reprojection_px
    ][: args.max_pairs]
    if not pairs:
        raise RuntimeError("no pair satisfies the configured reliability gate")
    device = torch.device("cuda:0")
    rows = []
    artifacts = {}
    for pair in pairs:
        source_np = np.load(args.teacher_root / records[pair["source"]]["geometry"])
        target_np = np.load(args.teacher_root / records[pair["target"]]["geometry"])
        source_rgb = torch.from_numpy(source_np["rgb"]).to(device)
        target_rgb = torch.from_numpy(target_np["rgb"]).to(device)
        source_float = source_rgb.float() / 255.0
        target_float = target_rgb.float() / 255.0
        rotation, _ = cv2.Rodrigues(np.asarray(pair["rvec"], dtype=np.float32))
        translation = np.asarray(pair["tvec"], dtype=np.float32)
        target_center_source = -rotation.T @ translation
        target_c2source_rotation = rotation.T
        resident = {
            "points": torch.from_numpy(source_np["points"]).to(device),
            "rgb": source_rgb,
            "mask": torch.from_numpy(source_np["mask"]).to(device),
            "intrinsics": torch.from_numpy(source_np["intrinsics"]).to(device),
            "center": torch.from_numpy(target_center_source).to(device),
            "rotation": torch.from_numpy(target_c2source_rotation).to(device),
        }
        for radius in (0, 1):
            warped_u8, _, mask = forward_splat_torch(
                resident["points"],
                resident["rgb"],
                resident["mask"],
                resident["intrinsics"],
                resident["center"],
                resident["rotation"],
                radius,
            )
            warped = warped_u8.float() / 255.0
            mask_float = mask.unsqueeze(-1).float()
            normalizer = (mask_float.sum() * 3).clamp_min(1.0)
            warp_valid_l1 = ((warped - target_float).abs() * mask_float).sum() / normalizer
            copy_valid_l1 = ((source_float - target_float).abs() * mask_float).sum() / normalizer
            composite = torch.where(mask.unsqueeze(-1), warped, source_float)
            row = {
                "pair": {
                    key: pair[key]
                    for key in (
                        "scene",
                        "source",
                        "target",
                        "inliers",
                        "inlier_ratio",
                        "median_reprojection_px",
                    )
                },
                "splat_radius": radius,
                "translation_norm_teacher_units": float(np.linalg.norm(translation)),
                "rotation_degrees": float(np.linalg.norm(pair["rvec"]) * 180.0 / np.pi),
                "coverage": float(mask_float.mean()),
                "warp_rgb_l1_valid": float(warp_valid_l1),
                "copy_rgb_l1_same_valid": float(copy_valid_l1),
                "copy_psnr_full": psnr(source_float, target_float),
                "warp_zero_holes_psnr_full": psnr(warped, target_float),
                "composite_psnr_full": psnr(composite, target_float),
                "copy_global_ssim_full": global_ssim(source_float, target_float),
                "composite_global_ssim_full": global_ssim(composite, target_float),
            }
            rows.append(row)
            key = (pair["source"], radius)
            artifacts[key] = {
                "source": source_np["rgb"],
                "target": target_np["rgb"],
                "warped": warped_u8.cpu().numpy(),
                "composite": (composite.clamp(0, 1) * 255).byte().cpu().numpy(),
                "mask": mask.cpu().numpy(),
                "delta": row["composite_psnr_full"] - row["copy_psnr_full"],
            }
    args.output.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output / "visuals"
    visual_dir.mkdir(exist_ok=True)
    for radius in (0, 1):
        candidates = sorted(
            ((key, value) for key, value in artifacts.items() if key[1] == radius),
            key=lambda item: item[1]["delta"],
        )
        selected = candidates[:2] + candidates[-2:]
        for rank, (key, item) in enumerate(selected):
            warped_holes = item["warped"].copy()
            warped_holes[~item["mask"]] = np.array([255, 0, 255], dtype=np.uint8)
            panel = np.concatenate(
                (item["source"], item["target"], warped_holes, item["composite"]), axis=1
            )
            filename = f"r{radius}_{rank}_{key[0]}_delta{item['delta']:+.3f}.png".replace("/", "_")
            cv2.imwrite(str(visual_dir / filename), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    report = {
        "schema_version": 1,
        "stage": "RGB pose/geometry diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "pose_status": "MoGe-point-assisted SIFT + PnP-RANSAC; not ground truth",
        "pair_count": len(pairs),
        "row_count": len(rows),
        "panel_order": ["source", "target", "warp_with_magenta_holes", "warp_plus_source_copy"],
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pair_count": len(pairs), "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
