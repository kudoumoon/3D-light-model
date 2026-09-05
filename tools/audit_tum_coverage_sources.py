#!/usr/bin/env python3
"""分解真实 TUM latent warp 的 coverage 来源，并测试零参数微孔闭合。

本脚本不训练或新增模型。它在严格测试场景上区分：输入深度有效率、
视野内投影率、GT-depth renderer coverage、Student coverage，以及局部闭合新增
区域相对 Copy baseline 的质量。只有新增区域优于 Copy 时，闭合才有工程价值。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_geometry_head import points_from_depth
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_reprojection_loss import forward_splat_latent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-scenes", nargs="+", default=("tum_xyz_test", "tum_rpy_test"))
    parser.add_argument("--depth-consistency-relative", type=float, default=0.10)
    parser.add_argument("--valid-threshold", type=float, default=0.50)
    return parser.parse_args()


def pose_matrix(pair: dict[str, object]) -> torch.Tensor:
    rotation, _ = cv2.Rodrigues(np.asarray(pair["rvec"], dtype=np.float32))
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = torch.from_numpy(rotation)
    transform[:3, 3] = torch.tensor(pair["tvec"], dtype=torch.float32)
    return transform[None]


def target_reconstructable_mask(
    target_depth: torch.Tensor,
    target_valid: torch.Tensor,
    source_depth: torch.Tensor,
    source_valid: torch.Tensor,
    intrinsics: torch.Tensor,
    source_to_target: torch.Tensor,
    relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由 target GT depth 反查 source，得到 FOV 上界与同表面可重投影上界。"""
    target_points = points_from_depth(target_depth, intrinsics).flatten(2).transpose(1, 2)
    target_to_source = torch.linalg.inv(source_to_target)
    source_points = (
        target_points @ target_to_source[:, :3, :3].transpose(1, 2)
        + target_to_source[:, None, :3, 3]
    )
    z = source_points[..., 2]
    u = intrinsics[:, 0, 0:1] * source_points[..., 0] / z.clamp_min(1e-6) + intrinsics[:, 0, 2:3]
    v = intrinsics[:, 1, 1:2] * source_points[..., 1] / z.clamp_min(1e-6) + intrinsics[:, 1, 2:3]
    inside = ((z > 1e-6) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).view_as(target_valid)
    grid = torch.stack((2 * u - 1, 2 * v - 1), dim=-1).view(
        target_depth.shape[0], target_depth.shape[2], target_depth.shape[3], 2
    )
    sampled_depth = F.grid_sample(source_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    sampled_valid = F.grid_sample(source_valid, grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5
    consistent = (sampled_depth - z.view_as(target_depth)).abs() <= relative_tolerance * z.view_as(target_depth).clamp_min(1e-3)
    fov = target_valid.bool() & inside
    reconstructable = fov & sampled_valid & consistent
    return fov, reconstructable


def masked_l1(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = mask.sum() * first.shape[1]
    if int(denominator) == 0:
        return float("nan")
    return float((((first - second).abs() * mask).sum() / denominator).detach().cpu())


def close_micro_holes(
    latent: torch.Tensor, valid: torch.Tensor, minimum_neighbors: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """单次 3x3 归一化邻域传播；无参数，只填紧邻已有投影的小孔。"""
    mask = valid.float()
    count = F.avg_pool2d(mask, 3, stride=1, padding=1) * 9
    summed = F.avg_pool2d(latent * mask, 3, stride=1, padding=1) * 9
    candidate = summed / count.clamp_min(1)
    newly_filled = (~valid) & (count >= minimum_neighbors)
    closed = torch.where(newly_filled.expand_as(latent), candidate, latent)
    return closed, valid | newly_filled, newly_filled


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


def overlap_ratio(first: torch.Tensor, second: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(((first.bool() & second.bool()).sum() / denominator.sum().clamp_min(1)).cpu())


@torch.inference_mode()
def main() -> None:
    args = arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one checked idle GPU with CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device("cuda:0")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
    pairs = [row for row in pairs if row.get("ok") and row["scene"] in set(args.test_scenes)]
    index = {sample_id: offset for offset, sample_id in enumerate(cache["sample_ids"])}
    if not pairs:
        raise RuntimeError("test pair split is empty")
    model = LatentGeometryHeadV2().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for pair in pairs:
        source_index, target_index = index[pair["source"]], index[pair["target"]]
        source = cache["latent"][source_index : source_index + 1].to(device)
        target = cache["latent"][target_index : target_index + 1].to(device)
        source_depth = cache["depth"][source_index : source_index + 1].to(device)
        target_depth = cache["depth"][target_index : target_index + 1].to(device)
        source_valid = cache["valid"][source_index : source_index + 1].to(device)
        target_valid = cache["valid"][target_index : target_index + 1].to(device)
        intrinsics = cache["intrinsics"][source_index : source_index + 1].to(device)
        transform = pose_matrix(pair).to(device)
        teacher = forward_splat_latent(
            source, points_from_depth(source_depth, intrinsics), source_valid, intrinsics, transform
        )
        output = model(source, intrinsics)
        predicted_valid = (output.latent_valid >= args.valid_threshold).float()
        student_gt_valid = forward_splat_latent(
            source, output.latent_points, source_valid, intrinsics, transform
        )
        student_pred_valid = forward_splat_latent(
            source, output.latent_points, predicted_valid, intrinsics, transform
        )
        fov, reconstructable = target_reconstructable_mask(
            target_depth, target_valid, source_depth, source_valid, intrinsics, transform,
            args.depth_consistency_relative,
        )
        student_mask = student_pred_valid.projected_valid
        if "dense_projected_displacement_px_median" in pair:
            motion_px = float(pair["dense_projected_displacement_px_median"])
        else:
            translation = float(np.linalg.norm(np.asarray(pair["tvec"], dtype=np.float32)))
            rotation, _ = cv2.Rodrigues(np.asarray(pair["rvec"], dtype=np.float32))
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
            median_depth = float(source_depth[source_valid.bool()].median().clamp_min(1e-3))
            focal_px = float(intrinsics[0, 0, 0] * source_depth.shape[-1])
            motion_px = focal_px * (translation / median_depth + angle)
        row = {
            "scene": pair["scene"],
            "source": pair["source"],
            "target": pair["target"],
            "frame_delta": abs(
                int(str(pair["target"]).rsplit("_", 1)[1])
                - int(str(pair["source"]).rsplit("_", 1)[1])
            ),
            "motion_px": motion_px,
            "source_depth_valid": float(source_valid.mean()),
            "target_depth_valid": float(target_valid.mean()),
            "target_in_source_fov": float(fov.float().mean()),
            "oracle_reconstructable": float(reconstructable.float().mean()),
            "teacher_renderer_coverage": float(teacher.coverage.mean()),
            "student_with_gt_valid_coverage": float(student_gt_valid.coverage.mean()),
            "student_inference_coverage": float(student_pred_valid.coverage.mean()),
            "student_reconstructable_recall": overlap_ratio(student_mask, reconstructable, reconstructable),
            "student_reconstructable_precision": overlap_ratio(student_mask, reconstructable, student_mask),
            "student_outside_fov_fraction": overlap_ratio(student_mask, ~fov, student_mask),
            "teacher_warp_l1": masked_l1(teacher.latent, target, teacher.projected_valid.float()),
            "teacher_copy_l1_same_support": masked_l1(source, target, teacher.projected_valid.float()),
            "student_warp_l1": masked_l1(student_pred_valid.latent, target, student_pred_valid.projected_valid.float()),
            "student_copy_l1_same_support": masked_l1(source, target, student_pred_valid.projected_valid.float()),
        }
        row["teacher_warp_beats_copy"] = float(
            row["teacher_warp_l1"] < row["teacher_copy_l1_same_support"]
        )
        row["student_warp_beats_copy"] = float(
            row["student_warp_l1"] < row["student_copy_l1_same_support"]
        )
        for minimum_neighbors in (1, 3, 5, 7):
            closed, closed_valid, added = close_micro_holes(
                student_pred_valid.latent, student_mask, minimum_neighbors
            )
            prefix = f"close_n{minimum_neighbors}"
            row[f"{prefix}_coverage"] = float(closed_valid.float().mean())
            row[f"{prefix}_added"] = float(added.float().mean())
            row[f"{prefix}_added_reconstructable_precision"] = overlap_ratio(
                added, reconstructable, added
            )
            row[f"{prefix}_added_l1"] = masked_l1(closed, target, added.float())
            row[f"{prefix}_added_copy_l1"] = masked_l1(source, target, added.float())
            row[f"{prefix}_full_support_l1"] = masked_l1(closed, target, closed_valid.float())
        rows.append(row)
    args.output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "stage": "coverage source decomposition and zero-parameter micro-hole closure",
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config": {
            "cache": str(args.cache), "pairs": str(args.pairs), "checkpoint": str(args.checkpoint),
            "test_scenes": args.test_scenes, "depth_consistency_relative": args.depth_consistency_relative,
            "valid_threshold": args.valid_threshold,
        },
        "pair_count": len(rows),
        "aggregate": aggregate(rows),
        "by_scene": {
            scene: aggregate([row for row in rows if row["scene"] == scene])
            for scene in sorted({row["scene"] for row in rows})
        },
        "by_frame_delta": {
            str(delta): aggregate([row for row in rows if row["frame_delta"] == delta])
            for delta in sorted({row["frame_delta"] for row in rows})
        },
        "motion_groups": {
            "lt50": aggregate([row for row in rows if row["motion_px"] < 50]),
            "50to100": aggregate([row for row in rows if 50 <= row["motion_px"] < 100]),
            "ge100": aggregate([row for row in rows if row["motion_px"] >= 100]),
        },
        "rows": rows,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "decision_rule": {
            "adopt_micro_closure_only_if": "added_region_l1 < added_region_copy_l1 and full-support quality remains acceptable",
            "new_trainable_parameters": 0,
            "public_shape_changed": False,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print("实验已完成")


if __name__ == "__main__":
    main()
