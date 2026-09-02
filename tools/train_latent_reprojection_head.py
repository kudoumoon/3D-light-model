#!/usr/bin/env python3
"""Fine-tune the fixed-shape latent geometry head with paired latent reprojection.

This is a mechanism study on scene-disjoint, estimated-pose Matrix-Game pairs.
It keeps the public M1 tensor contract unchanged and reports geometry and warp
metrics separately so a lower feature loss cannot hide geometry degradation.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_geometry_head import LatentGeometryHead, points_from_depth
from latent_reprojection_loss import compare_warp_to_copy, forward_splat_latent
from tools.train_latent_geometry_head import depth_gradient_loss, evaluate as evaluate_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warp-weight", type=float, default=1.0)
    parser.add_argument("--warp-mode", choices=("l1", "l1_cosine"), default="l1")
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--min-inliers", type=int, default=200)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--max-median-reprojection-px", type=float, default=1.5)
    parser.add_argument("--hard-motion-px", type=float, default=1.0)
    parser.add_argument("--validation-scenes", nargs="+", default=("game2_mid_right", "game3_mid_right"))
    parser.add_argument("--holdout-scenes", nargs="+", default=("game2_right", "game3_right"))
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


def pose_matrix(pair: dict[str, object]) -> torch.Tensor:
    rotation, _ = cv2.Rodrigues(np.asarray(pair["rvec"], dtype=np.float32))
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = torch.from_numpy(rotation)
    transform[:3, 3] = torch.tensor(pair["tvec"], dtype=torch.float32)
    return transform


class PairDataset(Dataset):
    def __init__(self, cache: dict[str, object], pairs: list[dict[str, object]]) -> None:
        self.cache = cache
        self.pairs = pairs
        self.index = {sample_id: index for index, sample_id in enumerate(cache["sample_ids"])}
        missing = sorted(
            {str(pair[key]) for pair in pairs for key in ("source", "target") if pair[key] not in self.index}
        )
        if missing:
            raise KeyError(f"pair samples absent from cache: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pair = self.pairs[index]
        source_index = self.index[str(pair["source"])]
        target_index = self.index[str(pair["target"])]
        return (
            self.cache["latent"][source_index],
            self.cache["latent"][target_index],
            self.cache["depth"][source_index],
            self.cache["valid"][source_index],
            self.cache["intrinsics"][source_index],
            pose_matrix(pair),
            torch.tensor(float(pair["dense_projected_displacement_px_median"])),
        )


def filter_pairs(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = json.loads(args.pairs.read_text())["pairs"]
    selected = [
        row
        for row in rows
        if row.get("ok")
        and row.get("inliers", 0) >= args.min_inliers
        and row.get("inlier_ratio", 0.0) >= args.min_inlier_ratio
        and row.get("median_reprojection_px", math.inf) <= args.max_median_reprojection_px
    ]
    if not selected:
        raise RuntimeError("no pair passes the preregistered reliability gate")
    return selected


def geometry_loss(
    output: object, target_depth: torch.Tensor, valid: torch.Tensor, edge_weight: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output.latent_depth.float()
    log_error = F.smooth_l1_loss(
        prediction.log(), target_depth.clamp_min(1e-6).log(), reduction="none"
    )
    depth = (log_error * valid).sum() / valid.sum().clamp_min(1)
    validity = F.binary_cross_entropy_with_logits(output.latent_valid_logits.float(), valid)
    edge = depth_gradient_loss(prediction, target_depth, valid)
    total = depth + 0.1 * validity + edge_weight * edge
    return total, {"depth": depth, "valid": validity, "edge": edge}


def feature_loss(
    warped: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    channels = target.shape[1]
    l1 = ((warped - target).abs() * mask).sum() / (mask.sum() * channels).clamp_min(1)
    cosine_map = 1.0 - F.cosine_similarity(warped, target, dim=1)
    cosine = (cosine_map * mask[:, 0]).sum() / mask[:, 0].sum().clamp_min(1)
    total = l1 if mode == "l1" else l1 + cosine
    return total, l1, cosine


def train_epoch(
    model: LatentGeometryHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for source, target, depth, valid, intrinsics, transform, _motion in loader:
        source, target, depth, valid, intrinsics, transform = (
            value.to(device) for value in (source, target, depth, valid, intrinsics, transform)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(source.to(torch.bfloat16), intrinsics)
        geometry, parts = geometry_loss(output, depth, valid, args.edge_weight)
        warp = forward_splat_latent(
            source, output.latent_points.float(), valid, intrinsics, transform
        )
        warp_loss, warp_l1, warp_cosine = feature_loss(
            warp.latent, target.float(), warp.projected_valid.float(), args.warp_mode
        )
        loss = args.geometry_weight * geometry + args.warp_weight * warp_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach()),
                "geometry": float(geometry.detach()),
                "depth": float(parts["depth"].detach()),
                "valid": float(parts["valid"].detach()),
                "edge": float(parts["edge"].detach()),
                "warp_l1": float(warp_l1.detach()),
                "warp_cosine": float(warp_cosine.detach()),
                "coverage": float(warp.coverage.mean().detach()),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.inference_mode()
def evaluate_pairs(
    model: LatentGeometryHead,
    loader: DataLoader,
    device: torch.device,
    hard_motion_px: float,
) -> dict[str, object]:
    model.eval()
    rows: list[dict[str, float]] = []
    for source, target, depth, valid, intrinsics, transform, motion in loader:
        source, target, depth, valid, intrinsics, transform = (
            value.to(device) for value in (source, target, depth, valid, intrinsics, transform)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(source.to(torch.bfloat16), intrinsics)
        warp = forward_splat_latent(
            source, output.latent_points.float(), valid, intrinsics, transform
        )
        comparison = compare_warp_to_copy(warp, source.float(), target.float())
        prediction = output.latent_depth.float()
        support = valid.bool()
        for batch_index in range(source.shape[0]):
            sample_support = support[batch_index]
            rows.append(
                {
                    "motion_px": float(motion[batch_index]),
                    "abs_rel": float(
                        (((prediction[batch_index] - depth[batch_index]).abs() / depth[batch_index].clamp_min(1e-6))[sample_support]).mean()
                    ),
                    "warp_l1": float(comparison["warp_valid_l1"]),
                    "copy_l1": float(comparison["copy_valid_l1"]),
                    "warp_cosine": float(comparison["warp_valid_cosine_similarity"]),
                    "copy_cosine": float(comparison["copy_valid_cosine_similarity"]),
                    "coverage": float(comparison["coverage"]),
                }
            )

    def aggregate(values: list[dict[str, float]]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        metrics = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
        metrics["count"] = len(values)
        metrics["warp_win_rate_l1"] = float(np.mean([row["warp_l1"] < row["copy_l1"] for row in values]))
        return metrics

    return {
        "all": aggregate(rows),
        "hard_motion": aggregate([row for row in rows if row["motion_px"] >= hard_motion_px]),
        "rows": rows,
    }


def geometry_loader(cache: dict[str, object], scenes: set[str], batch_size: int) -> DataLoader:
    indices = torch.tensor([index for index, scene in enumerate(cache["scenes"]) if scene in scenes])
    dataset = torch.utils.data.TensorDataset(
        cache["latent"][indices], cache["depth"][indices], cache["valid"][indices], cache["intrinsics"][indices]
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    if args.warp_weight < 0 or args.geometry_weight <= 0:
        raise ValueError("warp-weight must be non-negative and geometry-weight positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    pairs = filter_pairs(args)
    validation_scenes = set(args.validation_scenes)
    test_scenes = set(args.holdout_scenes)
    if validation_scenes & test_scenes:
        raise ValueError("validation and holdout scenes must be disjoint")
    train_pairs = [row for row in pairs if row["scene"] not in validation_scenes | test_scenes]
    validation_pairs = [row for row in pairs if row["scene"] in validation_scenes]
    test_pairs = [row for row in pairs if row["scene"] in test_scenes]
    if not train_pairs or not validation_pairs or not test_pairs:
        raise RuntimeError("scene-disjoint train/validation/test pair split is empty")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(cache, train_pairs), batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(PairDataset(cache, validation_pairs), batch_size=1, shuffle=False)
    test_loader = DataLoader(PairDataset(cache, test_pairs), batch_size=1, shuffle=False)
    model = LatentGeometryHead().to(device)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    initial_validation = evaluate_pairs(model, validation_loader, device, args.hard_motion_px)
    history: list[dict[str, object]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation_l1 = float(initial_validation["all"]["warp_l1"])
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        row: dict[str, object] = train_epoch(model, train_loader, optimizer, args, device)
        validation = evaluate_pairs(model, validation_loader, device, args.hard_motion_px)
        row["epoch"] = epoch + 1
        row["validation_warp_l1"] = validation["all"]["warp_l1"]
        row["validation_warp_win_rate"] = validation["all"]["warp_win_rate_l1"]
        history.append(row)
        if float(validation["all"]["warp_l1"]) < best_validation_l1:
            best_validation_l1 = float(validation["all"]["warp_l1"])
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    test_pairs_report = evaluate_pairs(model, test_loader, device, args.hard_motion_px)
    test_geometry = evaluate_geometry(
        model, geometry_loader(cache, test_scenes, args.batch_size), device
    )
    args.output.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": vars(args)}, args.output / "checkpoint.pt")
    report = {
        "schema_version": 1,
        "stage": "scene-disjoint estimated-pose latent-reprojection fine-tuning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "test_pairs": len(test_pairs),
            "validation_scenes": sorted(validation_scenes),
            "test_scenes": sorted(test_scenes),
            "pose_status": "MoGe-point-assisted SIFT + PnP-RANSAC; not ground truth",
        },
        "selection": {
            "metric": "validation pair warp_l1",
            "best_epoch": best_epoch,
            "initial_validation_warp_l1": initial_validation["all"]["warp_l1"],
            "best_validation_warp_l1": best_validation_l1,
        },
        "history": history,
        "test_pairs": test_pairs_report,
        "test_geometry": test_geometry,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "artifacts": {
            "cache_sha256": sha256(args.cache),
            "init_checkpoint_sha256": sha256(args.init_checkpoint),
            "pairs_sha256": sha256(args.pairs),
        },
        "output_contract": {
            "depth": "[B,1,H_l,W_l]",
            "points": "[B,3,H_l,W_l]",
            "valid": "[B,1,H_l,W_l]",
            "confidence": "[B,1,H_l,W_l] after frozen-geometry confidence head",
            "intrinsics": "[B,3,3]",
            "shape_changed": False,
        },
        "evidence_boundary": "Mechanism evidence on Matrix-Game generated frames with estimated poses; not real-data, GT-pose, or cross-VAE generalization evidence.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test_pairs": test_pairs_report, "test_geometry": test_geometry}, indent=2))


if __name__ == "__main__":
    main()
