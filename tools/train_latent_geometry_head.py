#!/usr/bin/env python3
"""Train the frozen-Wan latent-to-geometry student on cached teacher targets."""

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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_geometry_alignment import align_depth_to_latent
from latent_geometry_head import LatentGeometryHead, points_from_depth
from tools.run_latent3d_teacher_screen import prepare_rgb, transform_geometry
from vae import WanVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--projection-weight", type=float, default=0.0)
    parser.add_argument("--tvod-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument("--head-blocks", type=int, default=3)
    parser.add_argument("--holdout-scenes", nargs="+", default=("game2_right", "game3_right"))
    parser.add_argument(
        "--validation-scenes", nargs="+", default=("game2_mid_right", "game3_mid_right")
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


def build_cache(args: argparse.Namespace, device: torch.device) -> None:
    if args.cache.exists():
        return
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(
        device, torch.bfloat16
    ).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    latents, depths, valids, intrinsics, scenes, sample_ids = [], [], [], [], [], []
    started = time.perf_counter()
    for index, record in enumerate(manifest["records"]):
        geometry = np.load(args.teacher_root / record["geometry"])
        rgb, crop = prepare_rgb(geometry["rgb"])
        depth, valid, intrinsic = transform_geometry(
            geometry["depth"], geometry["mask"].astype(np.float32), geometry["intrinsics"], crop
        )
        target_depth, _ = align_depth_to_latent(depth, valid, (44, 80), "median")
        target_valid = (F.avg_pool2d(valid, kernel_size=8, stride=8) >= 0.5).float()
        with torch.inference_mode():
            latent = vae.encode(rgb.to(device, torch.bfloat16), device=device, **tiler)[:, :, 0]
        latents.append(latent.cpu().to(torch.bfloat16))
        depths.append(target_depth.to(torch.float32))
        valids.append(target_valid.to(torch.float32))
        intrinsics.append(intrinsic.to(torch.float32))
        scenes.append(record["scene"])
        sample_ids.append(record["sample_id"])
        if (index + 1) % 100 == 0:
            print(json.dumps({"cached": index + 1, "total": len(manifest["records"])}), flush=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent": torch.cat(latents),
            "depth": torch.cat(depths),
            "valid": torch.cat(valids),
            "intrinsics": torch.cat(intrinsics),
            "scenes": scenes,
            "sample_ids": sample_ids,
            "vae_sha256": sha256(args.vae_checkpoint),
            "elapsed_seconds": time.perf_counter() - started,
        },
        args.cache,
    )


def depth_gradient_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction.log(), target.clamp_min(1e-6).log()
    mask_x = valid[..., :, 1:] * valid[..., :, :-1]
    mask_y = valid[..., 1:, :] * valid[..., :-1, :]
    dx = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return (dx.abs() * mask_x).sum() / mask_x.sum().clamp_min(1) + (dy.abs() * mask_y).sum() / mask_y.sum().clamp_min(1)


def project(points: torch.Tensor, intrinsics: torch.Tensor, translation_x: float) -> tuple[torch.Tensor, torch.Tensor]:
    target = points.clone()
    target[:, 0] += translation_x
    z = target[:, 2].clamp_min(1e-6)
    u = intrinsics[:, 0, 0, None, None] * target[:, 0] / z + intrinsics[:, 0, 2, None, None]
    v = intrinsics[:, 1, 1, None, None] * target[:, 1] / z + intrinsics[:, 1, 2, None, None]
    coords = torch.stack((u, v), dim=1)
    inside = ((target[:, 2] > 1e-6) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).unsqueeze(1)
    return coords, inside.float()


def occupancy(coords: torch.Tensor, valid: torch.Tensor, height: int = 22, width: int = 40) -> torch.Tensor:
    batch = coords.shape[0]
    u, v = coords[:, 0].flatten(1), coords[:, 1].flatten(1)
    support = valid.flatten(1) * ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
    x = (u.clamp(0, 1) * (width - 1)).clamp(0, width - 1)
    y = (v.clamp(0, 1) * (height - 1)).clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=width - 1), (y0 + 1).clamp(max=height - 1)
    wx, wy = x - x0, y - y0
    output = torch.zeros(batch, height * width, device=coords.device)
    for xi, yi, weight in ((x0, y0, (1-wx)*(1-wy)), (x1, y0, wx*(1-wy)), (x0, y1, (1-wx)*wy), (x1, y1, wx*wy)):
        output.scatter_add_(1, yi * width + xi, support * weight)
    return (1 - torch.exp(-output)).reshape(batch, 1, height, width)


def compute_loss(model: LatentGeometryHead, batch: tuple[torch.Tensor, ...], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    latent, target_depth, valid, intrinsics = (value.to(device) for value in batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(latent.to(torch.bfloat16), intrinsics)
    prediction = output.latent_depth.float()
    valid_logits = output.latent_valid_logits.float()
    log_error = F.smooth_l1_loss(prediction.log(), target_depth.clamp_min(1e-6).log(), reduction="none")
    depth_loss = (log_error * valid).sum() / valid.sum().clamp_min(1)
    valid_loss = F.binary_cross_entropy_with_logits(valid_logits, valid)
    edge_loss = depth_gradient_loss(prediction, target_depth, valid)
    pred_coords, pred_inside = project(output.latent_points.float(), intrinsics, 0.04)
    target_points = points_from_depth(target_depth, intrinsics)
    target_coords, target_inside = project(target_points, intrinsics, 0.04)
    projection = (F.smooth_l1_loss(pred_coords, target_coords, reduction="none") * valid).sum() / (2 * valid.sum()).clamp_min(1)
    target_support = valid * target_inside
    tvod = F.smooth_l1_loss(occupancy(pred_coords, valid * pred_inside), occupancy(target_coords, target_support))
    loss = depth_loss + 0.1 * valid_loss + args.edge_weight * edge_loss + args.projection_weight * projection + args.tvod_weight * tvod
    return loss, {"loss": float(loss.detach()), "depth": float(depth_loss.detach()), "valid": float(valid_loss.detach()), "edge": float(edge_loss.detach()), "projection": float(projection.detach()), "tvod": float(tvod.detach())}


def evaluate(model: LatentGeometryHead, loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = {key: 0.0 for key in ("abs_rel", "log_rmse", "delta1", "projection_l1", "occupancy_l1", "valid_iou")}
    count = 0
    model.eval()
    with torch.inference_mode():
        for latent, target_depth, valid, intrinsics in loader:
            latent, target_depth, valid, intrinsics = (value.to(device) for value in (latent, target_depth, valid, intrinsics))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(latent.to(torch.bfloat16), intrinsics)
            prediction = output.latent_depth.float()
            support = valid.bool()
            ratio = torch.maximum(prediction / target_depth.clamp_min(1e-6), target_depth / prediction.clamp_min(1e-6))
            totals["abs_rel"] += float((((prediction-target_depth).abs()/target_depth.clamp_min(1e-6))[support]).mean())
            totals["log_rmse"] += float(torch.sqrt(((prediction.log()-target_depth.clamp_min(1e-6).log())[support]).square().mean()))
            totals["delta1"] += float((ratio[support] < 1.25).float().mean())
            pred_coords, pred_inside = project(output.latent_points.float(), intrinsics, 0.04)
            target_coords, target_inside = project(points_from_depth(target_depth, intrinsics), intrinsics, 0.04)
            totals["projection_l1"] += float(((pred_coords-target_coords).abs()*valid).sum()/(2*valid.sum()).clamp_min(1))
            totals["occupancy_l1"] += float(F.l1_loss(occupancy(pred_coords, valid*pred_inside), occupancy(target_coords, valid*target_inside)))
            pred_valid = output.latent_valid > 0.5
            totals["valid_iou"] += float((pred_valid & support).sum() / (pred_valid | support).sum().clamp_min(1))
            count += 1
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    build_cache(args, device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    holdout = set(args.holdout_scenes)
    validation_scenes = set(args.validation_scenes)
    if holdout & validation_scenes:
        raise ValueError("holdout and validation scenes must be disjoint")
    train_indices = torch.tensor(
        [
            index
            for index, scene in enumerate(cache["scenes"])
            if scene not in holdout and scene not in validation_scenes
        ]
    )
    validation_indices = torch.tensor(
        [index for index, scene in enumerate(cache["scenes"]) if scene in validation_scenes]
    )
    test_indices = torch.tensor(
        [index for index, scene in enumerate(cache["scenes"]) if scene in holdout]
    )
    tensors = (cache["latent"], cache["depth"], cache["valid"], cache["intrinsics"])
    train = TensorDataset(*(value[train_indices] for value in tensors))
    validation = TensorDataset(*(value[validation_indices] for value in tensors))
    test = TensorDataset(*(value[test_indices] for value in tensors))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    model = LatentGeometryHead(width=args.head_width, blocks=args.head_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    best_state = None
    best_validation = None
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rows = []
        for batch in train_loader:
            loss, row = compute_loss(model, batch, args, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows.append(row)
        epoch_row = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        epoch_row["epoch"] = epoch + 1
        validation_metrics = evaluate(model, validation_loader, device)
        epoch_row["validation_projection_l1"] = validation_metrics["projection_l1"]
        epoch_row["validation_abs_rel"] = validation_metrics["abs_rel"]
        history.append(epoch_row)
        if best_validation is None or validation_metrics["projection_l1"] < best_validation["projection_l1"]:
            best_validation = validation_metrics
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(epoch_row), flush=True)
    if best_state is None or best_validation is None or best_epoch is None:
        raise RuntimeError("validation checkpoint selection failed")
    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, device)
    torch.cuda.synchronize()
    probes = cache["latent"][test_indices[:1]].to(device, torch.bfloat16)
    probe_k = cache["intrinsics"][test_indices[:1]].to(device)
    for _ in range(100):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(probes, probe_k)
    torch.cuda.synchronize()
    timings = []
    for _ in range(1000):
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(probes, probe_k)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    args.output.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": vars(args), "metrics": metrics}, args.output / "checkpoint.pt")
    report = {
        "schema_version": 1,
        "stage": "scene-holdout latent geometry student",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": sha256(args.vae_checkpoint),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "total": len(cache["scenes"]),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "validation_scenes": sorted(validation_scenes),
            "holdout_scenes": sorted(holdout),
        },
        "selection": {
            "metric": "validation_projection_l1",
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "test": metrics,
        "latency": {"mean_ms": float(np.mean(timings)), "median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, 0.95)), "peak_allocated_bytes": torch.cuda.max_memory_allocated()},
        "runtime": {"training_seconds_excluding_cache": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(0)},
        "evidence_boundary": "MoGe-3 pseudo-label scene holdout on Matrix-Game generated frames; not real-data or cross-VAE generalization evidence",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"test": metrics, "latency": report["latency"]}, indent=2))


if __name__ == "__main__":
    main()
