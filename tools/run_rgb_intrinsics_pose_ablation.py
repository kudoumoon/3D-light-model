#!/usr/bin/env python3
"""Diagnose pseudo-pose sensitivity to frame-wise MoGe intrinsics.

This experiment reuses one SIFT 3D-2D correspondence set per pair, estimates
PnP independently under three explicit camera assumptions, and evaluates RGB
warp against Copy on exactly the same projected-valid support.
"""

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
    return parser.parse_args()


def normalized_to_pixel(k: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.diag([width, height, 1.0]).astype(np.float32)
    pixel = scale @ k.astype(np.float32)
    pixel[0, 2] -= 0.5
    pixel[1, 2] -= 0.5
    return pixel


def collect_correspondences(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    points: np.ndarray,
    geometry_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    source_key, source_desc = sift.detectAndCompute(
        cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY), None
    )
    target_key, target_desc = sift.detectAndCompute(
        cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY), None
    )
    if source_desc is None or target_desc is None:
        raise RuntimeError("SIFT did not find descriptors")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_desc, target_desc, k=2)
    matches = [
        candidates[0]
        for candidates in knn
        if len(candidates) == 2
        and candidates[0].distance < 0.75 * candidates[1].distance
    ]
    height, width = geometry_valid.shape
    object_points, image_points = [], []
    for match in matches:
        u, v = source_key[match.queryIdx].pt
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < width and 0 <= y < height and geometry_valid[y, x]):
            continue
        point = points[y, x]
        if np.isfinite(point).all() and point[2] > 0:
            object_points.append(point)
            image_points.append(target_key[match.trainIdx].pt)
    if len(object_points) < 12:
        raise RuntimeError(f"only {len(object_points)} valid 3D-2D matches")
    return (
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        {
            "source_keypoints": len(source_key),
            "target_keypoints": len(target_key),
            "ratio_matches": len(matches),
            "valid_correspondences": len(object_points),
        },
    )


def estimate_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics_pixel: np.ndarray,
) -> dict:
    cv2.setRNGSeed(7)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        intrinsics_pixel,
        None,
        iterationsCount=500,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 8:
        raise RuntimeError("PnP RANSAC failed")
    selected = inliers.ravel()
    refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
        object_points[selected],
        image_points[selected],
        intrinsics_pixel,
        None,
        rvec,
        tvec,
    )
    candidates = ((rvec, tvec), (refined_rvec, refined_tvec))
    evaluated = []
    for candidate_rvec, candidate_tvec in candidates:
        projected, _ = cv2.projectPoints(
            object_points[selected],
            candidate_rvec,
            candidate_tvec,
            intrinsics_pixel,
            None,
        )
        error = np.linalg.norm(projected[:, 0] - image_points[selected], axis=1)
        evaluated.append((float(np.median(error)), candidate_rvec, candidate_tvec, error))
    median_error, rvec, tvec, errors = min(evaluated, key=lambda item: item[0])
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "rotation": rotation.astype(np.float32),
        "translation": tvec.reshape(3).astype(np.float32),
        "inliers": int(len(selected)),
        "inlier_ratio": float(len(selected) / len(object_points)),
        "median_reprojection_px": median_error,
        "p95_reprojection_px": float(np.percentile(errors, 95)),
    }


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
    records = {record["sample_id"]: record for record in manifest["records"]}
    raw_pairs = json.loads(args.pairs.read_text())["pairs"]
    pairs = [
        pair
        for pair in raw_pairs
        if pair.get("ok")
        and pair.get("inliers", 0) >= 200
        and pair.get("inlier_ratio", 0.0) >= 0.6
        and pair.get("median_reprojection_px", float("inf")) <= 1.5
    ][: args.max_pairs]
    if not pairs:
        raise RuntimeError("no pair satisfies the configured reliability gate")

    pair_geometry = {}
    scene_intrinsics: dict[str, list[np.ndarray]] = {}
    for pair in pairs:
        source = np.load(args.teacher_root / records[pair["source"]]["geometry"])
        target = np.load(args.teacher_root / records[pair["target"]]["geometry"])
        pair_geometry[pair["source"]] = (source, target)
        scene_intrinsics.setdefault(pair["scene"], []).extend(
            (source["intrinsics"], target["intrinsics"])
        )
    fixed_intrinsics = {
        scene: np.median(np.stack(values), axis=0).astype(np.float32)
        for scene, values in scene_intrinsics.items()
    }

    device = torch.device("cuda:0")
    rows = []
    for pair in pairs:
        source, target = pair_geometry[pair["source"]]
        object_points, image_points, match_stats = collect_correspondences(
            source["rgb"], target["rgb"], source["points"], source["mask"]
        )
        camera_models = {
            "source_frame_k": source["intrinsics"].astype(np.float32),
            "target_frame_k_noncausal": target["intrinsics"].astype(np.float32),
            "scene_pair_median_k": fixed_intrinsics[pair["scene"]],
        }
        height, width = source["mask"].shape
        source_float = torch.from_numpy(source["rgb"]).to(device).float() / 255.0
        target_float = torch.from_numpy(target["rgb"]).to(device).float() / 255.0
        resident = {
            "points": torch.from_numpy(source["points"]).to(device),
            "rgb": torch.from_numpy(source["rgb"]).to(device),
            "mask": torch.from_numpy(source["mask"]).to(device),
        }
        for camera_model, intrinsics_normalized in camera_models.items():
            pose = estimate_pose(
                object_points,
                image_points,
                normalized_to_pixel(intrinsics_normalized, width, height),
            )
            rotation = pose.pop("rotation")
            translation = pose.pop("translation")
            target_c2source = rotation.T
            camera_center_source = -target_c2source @ translation
            warped_u8, _, mask = forward_splat_torch(
                resident["points"],
                resident["rgb"],
                resident["mask"],
                torch.from_numpy(intrinsics_normalized).to(device),
                torch.from_numpy(camera_center_source).to(device),
                torch.from_numpy(target_c2source).to(device),
                0,
            )
            warped = warped_u8.float() / 255.0
            valid = mask.unsqueeze(-1).float()
            normalizer = (valid.sum() * 3).clamp_min(1.0)
            composite = torch.where(mask.unsqueeze(-1), warped, source_float)
            rows.append(
                {
                    "pair": {key: pair[key] for key in ("scene", "source", "target")},
                    "camera_model": camera_model,
                    **match_stats,
                    **pose,
                    "translation_norm_teacher_units": float(np.linalg.norm(translation)),
                    "rotation_degrees": float(
                        np.degrees(
                            np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
                        )
                    ),
                    "coverage": float(mask.float().mean()),
                    "warp_rgb_l1_valid": float(
                        ((warped - target_float).abs() * valid).sum() / normalizer
                    ),
                    "copy_rgb_l1_same_valid": float(
                        ((source_float - target_float).abs() * valid).sum() / normalizer
                    ),
                    "copy_psnr_full": psnr(source_float, target_float),
                    "composite_psnr_full": psnr(composite, target_float),
                    "intrinsics_normalized": intrinsics_normalized.tolist(),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "stage": "RGB intrinsics and PnP sensitivity ablation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "gpu": torch.cuda.get_device_name(0),
        "pair_count": len(pairs),
        "camera_models": [
            "source_frame_k",
            "target_frame_k_noncausal",
            "scene_pair_median_k",
        ],
        "metric_protocol": "Warp and Copy L1 use exactly the same projected-valid mask; holes use source Copy only for the composite PSNR.",
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pair_count": len(pairs), "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
