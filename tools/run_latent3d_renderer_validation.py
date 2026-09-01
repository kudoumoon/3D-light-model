#!/usr/bin/env python3
"""CPU analytical validation for the latent-grid 3D renderer."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_geometry_head import points_from_depth
from latent_reprojection_loss import forward_splat_latent, latent_reprojection_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=44)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def intrinsics() -> torch.Tensor:
    return torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]])


def transform(tx: float = 0.0, ty: float = 0.0) -> torch.Tensor:
    value = torch.eye(4).unsqueeze(0)
    value[:, 0, 3] = tx
    value[:, 1, 3] = ty
    return value


def repository_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def main() -> None:
    args = parse_args()
    if min(args.height, args.width, args.channels) <= 0:
        raise ValueError("height, width and channels must be positive")
    torch.manual_seed(args.seed)
    feature_tolerance = 5e-5
    coverage_tolerance = 2e-6
    features = torch.randn(1, args.channels, args.height, args.width)
    depth = torch.ones(1, 1, args.height, args.width)
    valid = torch.ones_like(depth)
    points = points_from_depth(depth, intrinsics())
    checks = []

    identity = forward_splat_latent(
        features, points, valid, intrinsics(), transform()
    )
    identity_error = float((identity.latent - features).abs().max())
    checks.append(
        {
            "name": "identity",
            "max_feature_error": identity_error,
            "coverage": float(identity.coverage.mean()),
            "support_mass_mean": float(identity.support_mass.mean()),
            "passed": identity_error < feature_tolerance
            and float(identity.coverage.mean()) == 1.0
            and abs(float(identity.support_mass.mean()) - 1.0) < 1e-6,
        }
    )

    for cells in (1, 4, 8):
        shifted = forward_splat_latent(
            features,
            points,
            valid,
            intrinsics(),
            transform(tx=cells / args.width),
        )
        expected = features[..., : args.width - cells]
        actual = shifted.latent[..., cells:]
        expected_coverage = (args.width - cells) / args.width
        checks.append(
            {
                "name": f"horizontal_translation_{cells}_cells",
                "max_feature_error": float((actual - expected).abs().max()),
                "coverage": float(shifted.coverage.mean()),
                "expected_coverage": expected_coverage,
                "passed": float((actual - expected).abs().max()) < feature_tolerance
                and abs(float(shifted.coverage.mean()) - expected_coverage) < coverage_tolerance,
            }
        )

    vertical_cells = 5
    shifted_vertical = forward_splat_latent(
        features,
        points,
        valid,
        intrinsics(),
        transform(ty=vertical_cells / args.height),
    )
    vertical_error = float(
        (
            shifted_vertical.latent[..., vertical_cells:, :]
            - features[..., : args.height - vertical_cells, :]
        )
        .abs()
        .max()
    )
    expected_vertical_coverage = (args.height - vertical_cells) / args.height
    checks.append(
        {
            "name": "vertical_translation_5_cells",
            "max_feature_error": vertical_error,
            "coverage": float(shifted_vertical.coverage.mean()),
            "expected_coverage": expected_vertical_coverage,
            "passed": vertical_error < feature_tolerance
            and abs(float(shifted_vertical.coverage.mean()) - expected_vertical_coverage) < coverage_tolerance,
        }
    )

    collision_features = torch.tensor([[[[10.0, 100.0]]]])
    collision_points = torch.tensor(
        [[[[-0.25, -0.50]], [[0.0, 0.0]], [[1.0, 2.0]]]]
    )
    collision = forward_splat_latent(
        collision_features,
        collision_points,
        torch.ones(1, 1, 1, 2),
        intrinsics(),
        transform(),
        depth_temperature=12.0,
    )
    near_error = float((collision.latent[0, 0, 0, 0] - 10.0).abs())
    checks.append(
        {
            "name": "local_z_buffer_near_surface",
            "near_surface_error": near_error,
            "coverage": float(collision.coverage.mean()),
            "passed": near_error < 0.01
            and torch.equal(
                collision.projected_valid.flatten(), torch.tensor([True, False])
            ),
        }
    )

    scale = 7.5
    base_transform = transform(tx=2.25 / args.width)
    scaled_transform = base_transform.clone()
    scaled_transform[:, :3, 3] *= scale
    base = forward_splat_latent(
        features, points, valid, intrinsics(), base_transform
    )
    scaled = forward_splat_latent(
        features, points * scale, valid, intrinsics(), scaled_transform
    )
    scale_error = float((base.latent - scaled.latent).abs().max())
    checks.append(
        {
            "name": "joint_geometry_translation_scale_equivariance",
            "max_feature_error": scale_error,
            "mask_identical": bool(torch.equal(base.projected_valid, scaled.projected_valid)),
            "passed": scale_error < feature_tolerance
            and torch.equal(base.projected_valid, scaled.projected_valid),
        }
    )

    train_features = torch.randn(
        1, args.channels, args.height, args.width, requires_grad=True
    )
    train_depth = torch.ones(
        1, 1, args.height, args.width, requires_grad=True
    )
    train_points = points_from_depth(train_depth, intrinsics())
    train_warp = forward_splat_latent(
        train_features,
        train_points,
        valid,
        intrinsics(),
        transform(tx=0.4 / args.width),
    )
    loss = latent_reprojection_loss(
        train_warp, torch.zeros_like(train_features), mode="l1"
    )["loss"]
    loss.backward()
    feature_gradient_finite = bool(torch.isfinite(train_features.grad).all())
    depth_gradient_finite = bool(torch.isfinite(train_depth.grad).all())
    checks.append(
        {
            "name": "training_gradient",
            "feature_gradient_finite": feature_gradient_finite,
            "depth_gradient_finite": depth_gradient_finite,
            "feature_gradient_l1": float(train_features.grad.abs().sum()),
            "depth_gradient_l1": float(train_depth.grad.abs().sum()),
            "passed": feature_gradient_finite
            and depth_gradient_finite
            and float(train_features.grad.abs().sum()) > 0
            and float(train_depth.grad.abs().sum()) > 0,
        }
    )

    report = {
        "schema_version": 1,
        "stage": "P2 renderer analytical validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "config": vars(args)
        | {
            "output": str(args.output),
            "feature_tolerance": feature_tolerance,
            "coverage_tolerance": coverage_tolerance,
            "occupancy_threshold": 1e-3,
        },
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "check_count": len(checks)}, indent=2))
    if not report["passed"]:
        raise RuntimeError("renderer analytical validation failed; inspect metrics.json")


if __name__ == "__main__":
    main()
