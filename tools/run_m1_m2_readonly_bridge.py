#!/usr/bin/env python3
"""Read-only M1-to-M2 controlled integration and renderer attribution test."""

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

from latent_geometry_head import LatentGeometryOutput, points_from_depth
from latent_m2_bridge import export_latent_geometry_to_m2
from latent_reprojection_loss import compare_warp_to_copy, forward_splat_latent
from tools.run_latent3d_teacher_screen import prepare_rgb
from vae import WanVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-repo", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    return parser.parse_args()


def git_state(path: Path) -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shifted(value: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx < 0 or dy < 0 or dx >= value.shape[-1] or dy >= value.shape[-2]:
        raise ValueError("this controlled test accepts in-bounds right/down shifts only")
    result = torch.zeros_like(value)
    source_y = slice(0, value.shape[-2] - dy) if dy else slice(None)
    target_y = slice(dy, None) if dy else slice(None)
    source_x = slice(0, value.shape[-1] - dx) if dx else slice(None)
    target_x = slice(dx, None) if dx else slice(None)
    result[..., target_y, target_x] = value[..., source_y, source_x]
    return result


def decoded_psnr(vae: WanVAE, latent: torch.Tensor, target: torch.Tensor, device: torch.device, tiler: dict) -> float:
    with torch.inference_mode():
        decoded = vae.decode(latent.to(torch.bfloat16).unsqueeze(2), device=device, **tiler).float()
    mse = (decoded - target.cpu()).square().mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    before = git_state(args.m2_repo)
    if before["dirty"]:
        raise RuntimeError("M2 repository must be clean for a read-only integration run")

    sys.path.insert(0, str(args.m2_repo.resolve()))
    from geosparse.geometry import RelativePose, geometry_pose_candidate
    import geosparse.geometry as m2_geometry_module

    if not Path(m2_geometry_module.__file__).resolve().is_relative_to(args.m2_repo.resolve()):
        raise RuntimeError("geosparse.geometry was not imported from the requested M2 repository")

    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = manifest["records"]
    indices = np.linspace(0, len(records) - 1, args.samples, dtype=int)
    selected = [records[index] for index in indices]
    device = torch.device("cuda:0")
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(
        device, torch.bfloat16
    ).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    height, width = 44, 80
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
        device=device,
    )
    depth = torch.ones((1, 1, height, width), device=device)
    geometry_output = LatentGeometryOutput(
        latent_depth=depth,
        latent_points=points_from_depth(depth, intrinsics),
        latent_valid_logits=torch.full_like(depth, 20.0),
        latent_confidence_logits=torch.full_like(depth, 20.0),
        intrinsics=intrinsics,
        spatial_downsample=8,
        temporal_downsample=4,
    )
    payload = export_latent_geometry_to_m2(geometry_output)
    scenarios = (
        {"name": "horizontal_integer_cells", "total_dx": 24, "total_dy": 0},
        {"name": "horizontal_half_cells", "total_dx": 12, "total_dy": 0},
        {"name": "vertical_half_cells", "total_dx": 0, "total_dy": 12},
    )
    rows = []

    for record in selected:
        source_np = np.load(args.teacher_root / record["geometry"])
        source_rgb, _ = prepare_rgb(source_np["rgb"])
        source_rgb = source_rgb.to(device)
        with torch.inference_mode():
            source_latent = vae.encode(
                source_rgb.to(torch.bfloat16), device=device, **tiler
            )[:, :, 0].to(device)
        source_history = source_latent.float().cpu().numpy()[:, :, None]

        for scenario in scenarios:
            target_rgbs = []
            target_latents = []
            m1_composites = []
            for horizon in (4, 8, 12):
                fraction = horizon / 12.0
                dx = round(scenario["total_dx"] * fraction)
                dy = round(scenario["total_dy"] * fraction)
                target_rgb = shifted(source_rgb, dx, dy)
                with torch.inference_mode():
                    target_latent = vae.encode(
                        target_rgb.to(torch.bfloat16), device=device, **tiler
                    )[:, :, 0].to(device)
                transform = torch.eye(4, device=device).unsqueeze(0)
                transform[0, 0, 3] = (scenario["total_dx"] / 8.0 / width) * fraction
                transform[0, 1, 3] = (scenario["total_dy"] / 8.0 / height) * fraction
                warp = forward_splat_latent(
                    source_latent,
                    geometry_output.latent_points,
                    geometry_output.latent_valid,
                    intrinsics,
                    transform,
                )
                comparison = compare_warp_to_copy(warp, source_latent, target_latent)
                target_rgbs.append(target_rgb)
                target_latents.append(target_latent)
                m1_composites.append(comparison["composite"])

            target_stack = torch.stack(target_latents, dim=2)
            m1_stack = torch.stack(m1_composites, dim=2)
            translation = np.array(
                [scenario["total_dx"] / 8.0 / width, scenario["total_dy"] / 8.0 / height, 0.0],
                dtype=np.float32,
            )
            pose = RelativePose(
                rotation_source_to_target=np.eye(3, dtype=np.float32),
                translation_source_to_target=translation,
                confidence=1.0,
                metadata={"source": "controlled_exact_plane"},
            )
            candidate = geometry_pose_candidate(
                source_history,
                payload.geometry,
                pose,
                horizons=(4, 8, 12),
                reference_horizon_rgb_frames=12.0,
                name="m1_latent_geometry_readonly_bridge",
            )
            m2_stack = torch.from_numpy(candidate.latent).to(device)
            copy_stack = source_latent.unsqueeze(2).expand_as(target_stack)
            row = {
                "sample_id": record["sample_id"],
                "scenario": scenario["name"],
                "target_pixel_shift_at_h12": [scenario["total_dx"], scenario["total_dy"]],
                "copy_latent_l1_full": float((copy_stack - target_stack).abs().mean()),
                "m1_renderer_latent_l1_full": float((m1_stack - target_stack).abs().mean()),
                "m2_renderer_latent_l1_full": float((m2_stack - target_stack).abs().mean()),
                "m2_visible_token_fraction": float(candidate.visible_tokens.mean()),
                "m2_confidence_mean": float(candidate.confidence.mean()),
                "horizons": [],
            }
            for frame, horizon in enumerate((4, 8, 12)):
                row["horizons"].append(
                    {
                        "rgb_horizon": horizon,
                        "copy_decoded_psnr": decoded_psnr(
                            vae, source_latent, target_rgbs[frame], device, tiler
                        ),
                        "m1_renderer_decoded_psnr": decoded_psnr(
                            vae, m1_stack[:, :, frame], target_rgbs[frame], device, tiler
                        ),
                        "m2_renderer_decoded_psnr": decoded_psnr(
                            vae, m2_stack[:, :, frame], target_rgbs[frame], device, tiler
                        ),
                    }
                )
            rows.append(row)

    after = git_state(args.m2_repo)
    if after != before:
        raise RuntimeError("M2 repository changed during the supposedly read-only run")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "stage": "M1-to-M2 read-only controlled bridge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "m2_repository": str(args.m2_repo.resolve()),
        "m2_state_before_and_after": before,
        "m2_module": str(Path(m2_geometry_module.__file__).resolve()),
        "m1_bridge": payload.metadata,
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": sha256(args.vae_checkpoint),
        "gpu": torch.cuda.get_device_name(0),
        "temporal_limit": "each target latent is encoded independently; this isolates the spatial M1/M2 interface and is not a causal 3-frame Video-VAE rollout",
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(selected), "row_count": len(rows), "m2_commit": before["commit"]}, indent=2))


if __name__ == "__main__":
    main()
