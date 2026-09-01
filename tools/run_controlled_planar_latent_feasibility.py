#!/usr/bin/env python3
"""Controlled Frozen-VAE feasibility test with exact planar 3D motion.

An integer RGB translation is the image formation of a fronto-parallel plane
under a calibrated lateral camera translation.  This removes teacher geometry,
PnP, and dynamic-scene errors while still exercising the production latent
renderer with explicit depth, intrinsics, and a source-to-target transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_geometry_head import points_from_depth
from latent_reprojection_loss import compare_warp_to_copy, forward_splat_latent
from tools.run_latent3d_teacher_screen import prepare_rgb
from vae import WanVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--pixel-shifts", type=int, nargs="+", default=(8, 16, 32, 64))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shift_right(image: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0 or pixels >= image.shape[-1]:
        raise ValueError("pixels must lie within the image width")
    target = torch.zeros_like(image)
    target[..., pixels:] = image[..., :-pixels]
    return target


def psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(-10.0 * torch.log10((first - second).square().mean().clamp_min(1e-12)))


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = manifest["records"]
    if len(records) < args.samples:
        raise ValueError("requested more samples than the teacher manifest contains")
    # Spread samples through the manifest rather than selecting one scene prefix.
    indices = np.linspace(0, len(records) - 1, args.samples, dtype=int)
    selected = [records[index] for index in indices]

    device = torch.device("cuda:0")
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(
        device, torch.bfloat16
    ).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    latent_height, latent_width = 44, 80
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
        device=device,
    )
    depth = torch.ones((1, 1, latent_height, latent_width), device=device)
    valid = torch.ones_like(depth)
    points = points_from_depth(depth, intrinsics)
    rows = []

    for record in selected:
        geometry = np.load(args.teacher_root / record["geometry"])
        source_rgb, _ = prepare_rgb(geometry["rgb"])
        source_rgb = source_rgb.to(device)
        with torch.inference_mode():
            source_latent = vae.encode(
                source_rgb.to(torch.bfloat16), device=device, **tiler
            )[:, :, 0].to(device)
            decoded_copy = vae.decode(
                source_latent.unsqueeze(2), device=device, **tiler
            ).float().to(device)
        for pixel_shift in args.pixel_shifts:
            target_rgb = shift_right(source_rgb, pixel_shift)
            with torch.inference_mode():
                target_latent = vae.encode(
                    target_rgb.to(torch.bfloat16), device=device, **tiler
                )[:, :, 0].to(device)
            # Wan spatial compression is 8x. A positive target-camera X
            # translation moves source points right by the requested cells.
            latent_shift = pixel_shift / 8.0
            transform = torch.eye(4, device=device).unsqueeze(0)
            transform[0, 0, 3] = latent_shift / latent_width
            warp = forward_splat_latent(
                source_latent, points, valid, intrinsics, transform
            )
            comparison = compare_warp_to_copy(warp, source_latent, target_latent)
            with torch.inference_mode():
                decoded_composite = vae.decode(
                    comparison["composite"].to(torch.bfloat16).unsqueeze(2),
                    device=device,
                    **tiler,
                ).float().to(device)
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "pixel_shift": pixel_shift,
                    "latent_cell_shift": latent_shift,
                    "warp_latent_l1_valid": float(comparison["warp_valid_l1"]),
                    "copy_latent_l1_same_valid": float(comparison["copy_valid_l1"]),
                    "warp_latent_cosine_similarity_valid": float(
                        comparison["warp_valid_cosine_similarity"]
                    ),
                    "copy_latent_cosine_similarity_same_valid": float(
                        comparison["copy_valid_cosine_similarity"]
                    ),
                    "latent_coverage": float(comparison["coverage"]),
                    "copy_decoded_psnr": psnr(decoded_copy, target_rgb),
                    "composite_decoded_psnr": psnr(decoded_composite, target_rgb),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "stage": "controlled planar Frozen-VAE latent 3D feasibility",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "gpu": torch.cuda.get_device_name(0),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": sha256(args.vae_checkpoint),
        "spatial_compression": 8,
        "target_construction": "exact integer RGB translation with zero-fill; equivalent to a calibrated fronto-parallel plane under lateral motion",
        "sample_count": len(selected),
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(selected), "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
