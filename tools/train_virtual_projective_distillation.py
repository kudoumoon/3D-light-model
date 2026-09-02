#!/usr/bin/env python3
"""Distill teacher projective behaviour into the fixed-shape latent geometry head.

Known virtual 6DoF camera transforms remove estimated-pose noise. The teacher
and student warp the same frozen VAE latent; supervision is therefore about
3D transport behaviour rather than matching another RGB frame's appearance.
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_geometry_head import LatentGeometryHead, points_from_depth
from latent_reprojection_loss import forward_splat_latent
from tools.train_latent_geometry_head import depth_gradient_loss, evaluate as evaluate_geometry
from tools.train_latent_reprojection_head import PairDataset, evaluate_pairs, filter_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument("--coordinate-weight", type=float, default=0.0)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--max-yaw-degrees", type=float, default=5.0)
    parser.add_argument("--max-translation", type=float, default=0.08)
    parser.add_argument("--hard-motion-px", type=float, default=1.0)
    parser.add_argument("--min-inliers", type=int, default=200)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--max-median-reprojection-px", type=float, default=1.5)
    parser.add_argument("--validation-scenes", nargs="+", default=("game2_mid_right", "game3_mid_right"))
    parser.add_argument("--holdout-scenes", nargs="+", default=("game2_right", "game3_right"))
    return parser.parse_args()


def repository_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def virtual_transforms(
    batch: int,
    generator: torch.Generator,
    device: torch.device,
    max_yaw_degrees: float,
    max_translation: float,
) -> torch.Tensor:
    random = torch.rand(batch, 4, generator=generator)
    yaw = (random[:, 0] * 2 - 1) * math.radians(max_yaw_degrees)
    tx = (random[:, 1] * 2 - 1) * max_translation
    ty = (random[:, 2] * 2 - 1) * max_translation * 0.5
    tz = (random[:, 3] * 2 - 1) * max_translation * 0.25
    cosine, sine = yaw.cos(), yaw.sin()
    transform = torch.eye(4).repeat(batch, 1, 1)
    transform[:, 0, 0] = cosine
    transform[:, 0, 2] = sine
    transform[:, 2, 0] = -sine
    transform[:, 2, 2] = cosine
    transform[:, :3, 3] = torch.stack((tx, ty, tz), dim=1)
    return transform.to(device)


def projected_coordinates(
    points: torch.Tensor, intrinsics: torch.Tensor, transform: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = points.shape
    xyz = points.flatten(2).transpose(1, 2)
    target = xyz @ transform[:, :3, :3].transpose(1, 2) + transform[:, None, :3, 3]
    z = target[..., 2]
    u = intrinsics[:, 0, 0:1] * target[..., 0] / z.clamp_min(1e-6) + intrinsics[:, 0, 2:3]
    v = intrinsics[:, 1, 1:2] * target[..., 1] / z.clamp_min(1e-6) + intrinsics[:, 1, 2:3]
    coordinates = torch.stack((u, v), dim=1).reshape(batch, 2, height, width)
    inside = ((z > 1e-6) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
    return coordinates, inside.reshape(batch, 1, height, width).float()


def geometry_loss(
    output: object, target_depth: torch.Tensor, valid: torch.Tensor, edge_weight: float
) -> torch.Tensor:
    prediction = output.latent_depth.float()
    log_error = F.smooth_l1_loss(
        prediction.log(), target_depth.clamp_min(1e-6).log(), reduction="none"
    )
    depth = (log_error * valid).sum() / valid.sum().clamp_min(1)
    validity = F.binary_cross_entropy_with_logits(output.latent_valid_logits.float(), valid)
    edge = depth_gradient_loss(prediction, target_depth, valid)
    return depth + 0.1 * validity + edge_weight * edge


def projective_losses(
    latent: torch.Tensor,
    student_points: torch.Tensor,
    teacher_points: torch.Tensor,
    valid: torch.Tensor,
    intrinsics: torch.Tensor,
    transform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_coordinates, student_inside = projected_coordinates(student_points, intrinsics, transform)
    teacher_coordinates, teacher_inside = projected_coordinates(teacher_points, intrinsics, transform)
    coordinate_mask = valid * student_inside * teacher_inside
    coordinate = ((student_coordinates - teacher_coordinates).abs() * coordinate_mask).sum()
    coordinate = coordinate / (2 * coordinate_mask.sum()).clamp_min(1)
    student_warp = forward_splat_latent(latent, student_points, valid, intrinsics, transform)
    teacher_warp = forward_splat_latent(latent, teacher_points, valid, intrinsics, transform)
    common = (student_warp.projected_valid & teacher_warp.projected_valid).float()
    feature = ((student_warp.latent - teacher_warp.latent).abs() * common).sum()
    feature = feature / (common.sum() * latent.shape[1]).clamp_min(1)
    coverage_gap = (student_warp.coverage - teacher_warp.coverage).abs().mean()
    return feature, coordinate, coverage_gap


@torch.inference_mode()
def evaluate_virtual(
    model: LatentGeometryHead,
    loader: DataLoader,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    rows: list[tuple[float, float, float]] = []
    for latent, depth, valid, intrinsics in loader:
        latent, depth, valid, intrinsics = (
            value.to(device) for value in (latent, depth, valid, intrinsics)
        )
        transform = virtual_transforms(
            len(latent), generator, device, args.max_yaw_degrees, args.max_translation
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(latent.to(torch.bfloat16), intrinsics)
        losses = projective_losses(
            latent.float(), output.latent_points.float(), points_from_depth(depth, intrinsics),
            valid, intrinsics, transform
        )
        rows.append(tuple(float(value) for value in losses))
    values = np.asarray(rows)
    return {
        "feature_l1": float(values[:, 0].mean()),
        "coordinate_l1": float(values[:, 1].mean()),
        "coverage_gap": float(values[:, 2].mean()),
    }


def make_loader(
    cache: dict[str, object], scenes: set[str], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    indices = torch.tensor([index for index, scene in enumerate(cache["scenes"]) if scene in scenes])
    dataset = TensorDataset(
        cache["latent"][indices], cache["depth"][indices], cache["valid"][indices], cache["intrinsics"][indices]
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    if min(args.geometry_weight, args.feature_weight, args.coordinate_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    all_scenes = set(cache["scenes"])
    validation_scenes = set(args.validation_scenes)
    test_scenes = set(args.holdout_scenes)
    train_scenes = all_scenes - validation_scenes - test_scenes
    train_loader = make_loader(cache, train_scenes, args.batch_size, True, args.seed)
    validation_loader = make_loader(cache, validation_scenes, args.batch_size, False, args.seed)
    test_geometry_loader = make_loader(cache, test_scenes, args.batch_size, False, args.seed)
    pairs = filter_pairs(args)
    test_pairs = [row for row in pairs if row["scene"] in test_scenes]
    test_pair_loader = DataLoader(PairDataset(cache, test_pairs), batch_size=1, shuffle=False)
    model = LatentGeometryHead().to(device)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    initial_validation = evaluate_virtual(model, validation_loader, args.seed + 1000, args, device)
    best_score = initial_validation["feature_l1"] + initial_validation["coordinate_l1"]
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    train_generator = torch.Generator().manual_seed(args.seed + 2000)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_rows = []
        for latent, depth, valid, intrinsics in train_loader:
            latent, depth, valid, intrinsics = (
                value.to(device) for value in (latent, depth, valid, intrinsics)
            )
            transform = virtual_transforms(
                len(latent), train_generator, device, args.max_yaw_degrees, args.max_translation
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(latent.to(torch.bfloat16), intrinsics)
            geometry = geometry_loss(output, depth, valid, args.edge_weight)
            feature, coordinate, coverage_gap = projective_losses(
                latent.float(), output.latent_points.float(), points_from_depth(depth, intrinsics),
                valid, intrinsics, transform
            )
            loss = (
                args.geometry_weight * geometry
                + args.feature_weight * feature
                + args.coordinate_weight * coordinate
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_rows.append((float(loss.detach()), float(geometry.detach()), float(feature.detach()), float(coordinate.detach()), float(coverage_gap.detach())))
        validation = evaluate_virtual(model, validation_loader, args.seed + 1000, args, device)
        score = validation["feature_l1"] + validation["coordinate_l1"]
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean([value[0] for value in epoch_rows])),
            "geometry": float(np.mean([value[1] for value in epoch_rows])),
            "feature": float(np.mean([value[2] for value in epoch_rows])),
            "coordinate": float(np.mean([value[3] for value in epoch_rows])),
            "coverage_gap": float(np.mean([value[4] for value in epoch_rows])),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    report = {
        "schema_version": 1,
        "stage": "virtual 6DoF latent projective behaviour distillation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "train_scenes": sorted(train_scenes),
            "validation_scenes": sorted(validation_scenes),
            "test_scenes": sorted(test_scenes),
            "test_pair_count": len(test_pairs),
        },
        "selection": {
            "metric": "validation_feature_l1 + validation_coordinate_l1",
            "best_epoch": best_epoch,
            "initial": initial_validation,
            "best_score": best_score,
        },
        "history": history,
        "test_virtual": evaluate_virtual(model, test_geometry_loader, args.seed + 3000, args, device),
        "test_geometry": evaluate_geometry(model, test_geometry_loader, device),
        "test_estimated_pose_pairs": evaluate_pairs(model, test_pair_loader, device, args.hard_motion_px),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "output_contract": {
            "depth": "[B,1,H_l,W_l]", "points": "[B,3,H_l,W_l]",
            "valid": "[B,1,H_l,W_l]", "confidence": "[B,1,H_l,W_l] after separate head",
            "intrinsics": "[B,3,3]", "shape_changed": False,
        },
        "evidence_boundary": "Virtual-pose projective mechanism test; held-out frame-pair poses remain estimated, not GT.",
    }
    args.output.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": vars(args)}, args.output / "checkpoint.pt")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selection", "test_virtual", "test_geometry", "test_estimated_pose_pairs")}, indent=2))


if __name__ == "__main__":
    main()
