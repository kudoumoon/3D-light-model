#!/usr/bin/env python3
"""Measure the causal RGB-frame attribution of the frozen Matrix-Game Wan VAE."""

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

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_chunk_geometry import wan_causal_rgb_groups
from tools.run_latent3d_teacher_screen import prepare_rgb
from vae import WanVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--rgb-frames", type=int, default=13)
    parser.add_argument(
        "--video-mode", choices=("repeated", "two_surface_motion"), default="repeated"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def perturb(video: torch.Tensor, frame: int, kind: str) -> torch.Tensor:
    result = video.clone()
    if kind == "blackout":
        result[:, :, frame] = 0
    elif kind == "local_invert":
        height, width = result.shape[-2:]
        y0, y1 = height // 4, 3 * height // 4
        x0, x1 = width // 4, 3 * width // 4
        result[:, :, frame, y0:y1, x0:x1] *= -1
    else:
        raise ValueError(f"unsupported perturbation: {kind}")
    return result


def shift_right(value: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels < 0 or pixels >= value.shape[-1]:
        raise ValueError("shift must be non-negative and remain in bounds")
    if pixels == 0:
        return value.clone()
    result = torch.zeros_like(value)
    result[..., pixels:] = value[..., :-pixels]
    return result


def make_two_surface_video(single: torch.Tensor, frames: int) -> torch.Tensor:
    _, _, _, height, width = single.shape
    mask = torch.zeros_like(single[:, :1])
    mask[..., height // 4 : 3 * height // 4, width // 4 : width // 2] = 1
    foreground = -torch.flip(single, dims=(-1,)) * mask
    video = []
    for frame in range(frames):
        background_shift = 2 * frame
        foreground_shift = 4 * frame
        shifted_mask = shift_right(mask, foreground_shift)
        shifted_foreground = shift_right(foreground, foreground_shift)
        shifted_background = shift_right(single, background_shift)
        video.append(shifted_background * (1 - shifted_mask) + shifted_foreground)
    return torch.cat(video, dim=2)


def psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = (first - second).float().square().mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def save_heatmap(matrix: np.ndarray, output: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 3.4))
    image = axis.imshow(matrix.T, aspect="auto", cmap="magma")
    axis.set_xlabel("Perturbed RGB frame")
    axis.set_ylabel("Affected latent slice")
    axis.set_title(title)
    axis.set_xticks(range(matrix.shape[0]))
    axis.set_yticks(range(matrix.shape[1]))
    figure.colorbar(image, ax=axis, label="normalized latent L1 influence")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    groups = wan_causal_rgb_groups(args.rgb_frames)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {args.output}")
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = manifest["records"]
    if args.samples <= 0 or len(records) < args.samples:
        raise ValueError("samples must be positive and available in the manifest")
    selected = [records[index] for index in np.linspace(0, len(records) - 1, args.samples, dtype=int)]

    device = torch.device("cuda:0")
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(
        device, torch.bfloat16
    ).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    kinds = ("blackout", "local_invert")
    rows: list[dict[str, object]] = []
    aggregate: dict[str, list[np.ndarray]] = {kind: [] for kind in kinds}

    for record in selected:
        geometry = np.load(args.teacher_root / record["geometry"])
        single, _ = prepare_rgb(geometry["rgb"])
        if args.video_mode == "repeated":
            video = single.repeat(1, 1, args.rgb_frames, 1, 1)
        else:
            video = make_two_surface_video(single, args.rgb_frames)
        video = video.to(device)
        with torch.inference_mode():
            baseline = vae.encode(video.to(torch.bfloat16), device=device, **tiler).to(device)
            decoded = vae.decode(baseline, device=device, **tiler).float().to(device)
        if baseline.shape[2] != len(groups):
            raise RuntimeError("observed latent time does not match the audited Wan grouping")

        independent_candidates = []
        with torch.inference_mode():
            for latent_index, group in enumerate(groups):
                group_deltas = []
                for rgb_index in group:
                    independent = vae.encode(
                        video[:, :, rgb_index : rgb_index + 1].to(torch.bfloat16),
                        device=device,
                        **tiler,
                    ).to(device)
                    group_deltas.append(
                        float((independent[:, :, 0] - baseline[:, :, latent_index]).abs().mean())
                    )
                independent_candidates.append(group_deltas)

        for kind in kinds:
            matrix = np.zeros((args.rgb_frames, len(groups)), dtype=np.float64)
            for frame in range(args.rgb_frames):
                with torch.inference_mode():
                    changed = vae.encode(
                        perturb(video, frame, kind).to(torch.bfloat16),
                        device=device,
                        **tiler,
                    ).to(device)
                matrix[frame] = (
                    changed - baseline
                ).abs().mean(dim=(0, 1, 3, 4)).float().cpu().numpy()
            normalizer = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, normalizer, out=np.zeros_like(matrix), where=normalizer > 0)
            aggregate[kind].append(normalized)
            for frame in range(args.rgb_frames):
                owner = next(index for index, group in enumerate(groups) if frame in group)
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "perturbation": kind,
                        "rgb_frame": frame,
                        "owner_latent": owner,
                        "dominant_latent": int(normalized[frame].argmax()),
                        "owner_fraction": float(normalized[frame, owner]),
                        "earlier_fraction": float(normalized[frame, :owner].sum()),
                        "future_fraction": float(normalized[frame, owner + 1 :].sum()),
                        "raw_l1_by_latent": matrix[frame].tolist(),
                    }
                )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "kind": "sequence_summary",
                "latent_shape": list(baseline.shape),
                "roundtrip_psnr": psnr(decoded, video),
                "independent_candidate_l1_by_latent": independent_candidates,
                "independent_anchor_l1_by_latent": [values[-1] for values in independent_candidates],
            }
        )

    args.output.mkdir(parents=True)
    aggregate_metrics: dict[str, object] = {}
    for kind, matrices in aggregate.items():
        mean_matrix = np.mean(matrices, axis=0)
        save_heatmap(
            mean_matrix,
            args.output / f"attribution_{kind}.png",
            f"Frozen Wan VAE causal attribution: {kind}",
        )
        kind_rows = [row for row in rows if row.get("perturbation") == kind]
        within_group = []
        sample_ids = sorted({row["sample_id"] for row in kind_rows})
        for sample_id in sample_ids:
            for latent_index, group in enumerate(groups):
                if len(group) == 1:
                    continue
                group_rows = {
                    row["rgb_frame"]: row
                    for row in kind_rows
                    if row["sample_id"] == sample_id and row["rgb_frame"] in group
                }
                influence = np.asarray(
                    [group_rows[frame]["raw_l1_by_latent"][latent_index] for frame in group]
                )
                within_group.append(influence / influence.sum())
        mean_within_group = np.mean(within_group, axis=0)
        aggregate_metrics[kind] = {
            "mean_owner_fraction": float(np.mean([row["owner_fraction"] for row in kind_rows])),
            "max_earlier_fraction": float(max(row["earlier_fraction"] for row in kind_rows)),
            "mean_future_fraction": float(np.mean([row["future_fraction"] for row in kind_rows])),
            "dominant_owner_rate": float(np.mean([row["owner_latent"] == row["dominant_latent"] for row in kind_rows])),
            "within_group_frame_weights_mean": mean_within_group.tolist(),
            "within_group_dominant_position": int(mean_within_group.argmax()),
            "within_group_frame_weights_std": np.std(within_group, axis=0, ddof=1).tolist(),
            "mean_matrix": mean_matrix.tolist(),
        }

    report = {
        "schema_version": 1,
        "stage": "Frozen Wan VAE causal temporal attribution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "visible_device_count": torch.cuda.device_count(),
        },
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": sha256(args.vae_checkpoint),
        "config": {
            "samples": args.samples,
            "rgb_frames": args.rgb_frames,
            "video_mode": args.video_mode,
            "causal_groups": groups,
            "perturbations": kinds,
            "tiler": tiler,
        },
        "aggregate": aggregate_metrics,
        "rows": rows,
        "evidence_boundary": "controlled VAE mechanism audit on demo-image textures; not a real-video temporal generalization result",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_metrics, indent=2))


if __name__ == "__main__":
    main()
