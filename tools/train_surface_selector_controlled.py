#!/usr/bin/env python3
"""Controlled two-surface test for reprojection-optimal latent geometry alignment.

The source image contains a near foreground patch over a far background.  Both
layers are shifted according to exact disparity, creating a target with known
geometry and camera motion.  This is a mechanism experiment, not evidence of
real-world generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_geometry_alignment import align_depth_to_latent
from latent_geometry_head import points_from_depth
from latent_reprojection_loss import compare_warp_to_copy, forward_splat_latent
from latent_surface_alignment import ReprojectionOptimalSurfaceSelector
from tools.run_latent3d_teacher_screen import prepare_rgb
from vae import WanVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--low-confidence-outliers", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shift_right(value: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0 or pixels >= value.shape[-1]:
        raise ValueError("shift must be positive and remain in bounds")
    result = torch.zeros_like(value)
    result[..., pixels:] = value[..., :-pixels]
    return result


def make_case(
    base: torch.Tensor, index: int, low_confidence_outliers: bool
) -> dict[str, torch.Tensor | int | float]:
    _, _, _, height, width = base.shape
    background_shift = (4, 8, 12, 16)[index % 4]
    foreground_depth = (0.5, 0.625, 0.4)[index % 3]
    foreground_shift = int(round(background_shift / foreground_depth))
    rect_height = (33, 57, 81, 19)[index % 4]
    rect_width = (47, 93, 129, 21)[(index * 3) % 4]
    top = 19 + (index * 37) % (height - rect_height - 38)
    left = 23 + (index * 53) % (width - rect_width - foreground_shift - 46)

    mask = torch.zeros_like(base[:, :1])
    mask[..., top : top + rect_height, left : left + rect_width] = 1
    foreground = torch.flip(base, dims=(-1,)) * -1
    source = base * (1 - mask) + foreground * mask
    true_depth = torch.ones_like(base[:, :1])
    true_depth = true_depth * (1 - mask) + foreground_depth * mask
    observed_depth = true_depth.clone()
    confidence = torch.ones_like(true_depth)
    if low_confidence_outliers:
        row_grid = torch.arange(height, device=base.device).view(1, 1, 1, height, 1)
        col_grid = torch.arange(width, device=base.device).view(1, 1, 1, 1, width)
        outliers = ((row_grid * 17 + col_grid * 29 + index) % 113 == 0).to(base.dtype)
        observed_depth = observed_depth * (1 - outliers) + 0.2 * outliers
        confidence = confidence * (1 - outliers) + 0.02 * outliers
    valid = torch.ones_like(true_depth)

    target_background = shift_right(base, background_shift)
    target_mask = shift_right(mask, foreground_shift)
    target_foreground = shift_right(foreground * mask, foreground_shift)
    target = target_background * (1 - target_mask) + target_foreground
    return {
        "source_rgb": source,
        "target_rgb": target,
        "depth": observed_depth[:, :, 0],
        "true_depth": true_depth[:, :, 0],
        "confidence": confidence[:, :, 0],
        "valid": valid[:, :, 0],
        "target_mask": target_mask[:, :, 0],
        "background_shift": background_shift,
        "foreground_shift": foreground_shift,
        "foreground_depth": foreground_depth,
    }


def psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = (first - second).float().square().mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def repository_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    if min(args.train_samples, args.test_samples, args.steps) <= 0:
        raise ValueError("sample counts and steps must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = manifest["records"]
    total = args.train_samples + args.test_samples
    selected = [records[index] for index in np.linspace(0, len(records) - 1, total, dtype=int)]
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(
        device, torch.bfloat16
    ).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
        device=device,
    )

    cases: list[dict[str, object]] = []
    for index, record in enumerate(selected):
        geometry = np.load(args.teacher_root / record["geometry"])
        base, _ = prepare_rgb(geometry["rgb"])
        case = make_case(base.to(device), index + args.seed, args.low_confidence_outliers)
        with torch.inference_mode():
            encoded_source = vae.encode(
                case["source_rgb"].to(torch.bfloat16), device=device, **tiler
            )[:, :, 0].float().to(device)
            encoded_target = vae.encode(
                case["target_rgb"].to(torch.bfloat16), device=device, **tiler
            )[:, :, 0].float().to(device)
        # The frozen encoder runs under inference_mode, but the selector needs
        # ordinary immutable tensors as inputs to its autograd graph.
        source_latent = encoded_source.clone()
        target_latent = encoded_target.clone()
        target_mask = case["target_mask"]
        dilated = F.max_pool2d(target_mask, kernel_size=9, stride=1, padding=4)
        eroded = -F.max_pool2d(-target_mask, kernel_size=9, stride=1, padding=4)
        target_boundary = F.adaptive_max_pool2d((dilated - eroded > 0).float(), (44, 80))
        case.update(
            {
                "sample_id": record["sample_id"],
                "source_latent": source_latent,
                "target_latent": target_latent,
                "target_boundary": target_boundary,
            }
        )
        cases.append(case)

    selector = ReprojectionOptimalSurfaceSelector(
        latent_channels=16, hidden=32, temperature=0.12
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=5e-4, weight_decay=1e-4)
    train_cases = cases[: args.train_samples]
    loss_curve = []
    for step in range(args.steps):
        case_losses = []
        for case in train_cases:
            aligned = selector(
                case["depth"], case["valid"], case["source_latent"], case["confidence"]
            )
            transform = torch.eye(4, device=device).unsqueeze(0)
            transform[:, 0, 3] = case["background_shift"] / 640.0
            warp = forward_splat_latent(
                case["source_latent"],
                points_from_depth(aligned.depth, intrinsics),
                aligned.valid,
                intrinsics,
                transform,
            )
            difference = (warp.latent - case["target_latent"]).abs().mean(dim=1, keepdim=True)
            projected = warp.projected_valid.float()
            global_l1 = (difference * projected).sum() / projected.sum().clamp_min(1)
            boundary_support = projected * case["target_boundary"]
            boundary_l1 = (difference * boundary_support).sum() / boundary_support.sum().clamp_min(1)
            case_losses.append(global_l1 + 1.5 * boundary_l1 + 1e-4 * aligned.ambiguity.mean())
        loss = torch.stack(case_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.detach()))

    methods = (
        "center", "average", "median", "minimum", "confidence_weighted", "learned_surface"
    )
    test_rows: list[dict[str, object]] = []
    selector.eval()
    with torch.inference_mode():
        for case in cases[args.train_samples :]:
            for method in methods:
                if method == "learned_surface":
                    selected_alignment = selector(
                        case["depth"], case["valid"], case["source_latent"], case["confidence"]
                    )
                    aligned_depth = selected_alignment.depth
                    aligned_valid = selected_alignment.valid
                    ambiguity = float(selected_alignment.ambiguity.mean())
                    dispersion = float(selected_alignment.relative_dispersion.mean())
                else:
                    aligned_depth, aligned_valid = align_depth_to_latent(
                        case["depth"],
                        case["valid"],
                        (44, 80),
                        method,
                        confidence=case["confidence"] if method == "confidence_weighted" else None,
                    )
                    ambiguity = None
                    dispersion = None
                transform = torch.eye(4, device=device).unsqueeze(0)
                transform[:, 0, 3] = case["background_shift"] / 640.0
                warp = forward_splat_latent(
                    case["source_latent"],
                    points_from_depth(aligned_depth, intrinsics),
                    aligned_valid,
                    intrinsics,
                    transform,
                )
                comparison = compare_warp_to_copy(
                    warp, case["source_latent"], case["target_latent"]
                )
                decoded = vae.decode(
                    comparison["composite"].to(torch.bfloat16).unsqueeze(2),
                    device=device,
                    **tiler,
                ).float().to(device)
                difference = (warp.latent - case["target_latent"]).abs().mean(dim=1, keepdim=True)
                boundary_support = warp.projected_valid.float() * case["target_boundary"]
                boundary_l1 = (difference * boundary_support).sum() / boundary_support.sum().clamp_min(1)
                test_rows.append(
                    {
                        "sample_id": case["sample_id"],
                        "method": method,
                        "background_shift_px": case["background_shift"],
                        "foreground_shift_px": case["foreground_shift"],
                        "foreground_depth": case["foreground_depth"],
                        "warp_valid_l1": float(comparison["warp_valid_l1"]),
                        "boundary_latent_l1": float(boundary_l1),
                        "copy_valid_l1": float(comparison["copy_valid_l1"]),
                        "warp_valid_cosine": float(comparison["warp_valid_cosine_similarity"]),
                        "coverage": float(comparison["coverage"]),
                        "decoded_composite_psnr": psnr(decoded, case["target_rgb"]),
                        "ambiguity_mean": ambiguity,
                        "relative_dispersion_mean": dispersion,
                    }
                )

    aggregates: dict[str, object] = {}
    for method in methods:
        subset = [row for row in test_rows if row["method"] == method]
        aggregates[method] = {
            "test_cases": len(subset),
            "warp_valid_l1_mean": float(np.mean([row["warp_valid_l1"] for row in subset])),
            "boundary_latent_l1_mean": float(np.mean([row["boundary_latent_l1"] for row in subset])),
            "copy_valid_l1_mean": float(np.mean([row["copy_valid_l1"] for row in subset])),
            "warp_l1_win_rate_vs_copy": float(np.mean([row["warp_valid_l1"] < row["copy_valid_l1"] for row in subset])),
            "decoded_composite_psnr_mean": float(np.mean([row["decoded_composite_psnr"] for row in subset])),
            "coverage_mean": float(np.mean([row["coverage"] for row in subset])),
        }

    args.output.mkdir(parents=True)
    torch.save(
        {"model": selector.state_dict(), "seed": args.seed, "steps": args.steps},
        args.output / "checkpoint.pt",
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.bar(
        list(methods),
        [aggregates[method]["warp_valid_l1_mean"] for method in methods],
    )
    axis.set_ylabel("Held-out latent L1 (lower is better)")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(args.output / "heldout_latent_l1.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.plot(loss_curve)
    axis.set_xlabel("training step")
    axis.set_ylabel("latent warp objective")
    figure.tight_layout()
    figure.savefig(args.output / "training_curve.png", dpi=180)
    plt.close(figure)

    report = {
        "schema_version": 2,
        "stage": "boundary-aware controlled two-surface reprojection-optimal alignment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "gpu": {"name": torch.cuda.get_device_name(0), "visible_device_count": torch.cuda.device_count()},
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": sha256(args.vae_checkpoint),
        "config": vars(args) | {"teacher_root": str(args.teacher_root), "vae_checkpoint": str(args.vae_checkpoint), "output": str(args.output)},
        "parameter_count": sum(parameter.numel() for parameter in selector.parameters()),
        "training": {
            "initial_loss": loss_curve[0],
            "final_loss": loss_curve[-1],
            "last_50_mean": float(np.mean(loss_curve[-50:])),
        },
        "aggregates": aggregates,
        "rows": test_rows,
        "evidence_boundary": "controlled two-surface mechanism study on demo-image textures; not real-data generalization evidence",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"training": report["training"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
