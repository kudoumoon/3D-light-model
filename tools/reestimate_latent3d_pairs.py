#!/usr/bin/env python3
"""Re-estimate screened source-to-target poses with an explicit pixel K."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_rgb_intrinsics_pose_ablation import (
    collect_correspondences,
    estimate_pose,
    normalized_to_pixel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--input-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=25)
    return parser.parse_args()


def dense_displacement(
    points: np.ndarray,
    mask: np.ndarray,
    intrinsics_normalized: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, float]:
    height, width = mask.shape
    yy, xx = np.nonzero(mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0))
    xyz = points[yy, xx]
    target = xyz @ rotation.T + translation[None]
    front = np.isfinite(target).all(axis=-1) & (target[:, 2] > 1e-4)
    target, xx, yy = target[front], xx[front], yy[front]
    xy = target[:, :2] / target[:, 2:3]
    u = (intrinsics_normalized[0, 0] * xy[:, 0] + intrinsics_normalized[0, 2]) * width - 0.5
    v = (intrinsics_normalized[1, 1] * xy[:, 1] + intrinsics_normalized[1, 2]) * height - 0.5
    displacement = np.sqrt(np.square(u - xx) + np.square(v - yy))
    return {
        "dense_projected_displacement_px_median": float(np.median(displacement)),
        "dense_projected_displacement_px_p95": float(np.percentile(displacement, 95)),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = {record["sample_id"]: record for record in manifest["records"]}
    raw_pairs = json.loads(args.input_pairs.read_text())["pairs"]
    selected = [
        pair
        for pair in raw_pairs
        if pair.get("ok")
        and pair.get("inliers", 0) >= 200
        and pair.get("inlier_ratio", 0.0) >= 0.6
        and pair.get("median_reprojection_px", float("inf")) <= 1.5
    ][: args.max_pairs]
    rows = []
    for old in selected:
        source = np.load(args.teacher_root / records[old["source"]]["geometry"])
        target = np.load(args.teacher_root / records[old["target"]]["geometry"])
        object_points, image_points, match_stats = collect_correspondences(
            source["rgb"], target["rgb"], source["points"], source["mask"]
        )
        height, width = source["mask"].shape
        pose = estimate_pose(
            object_points,
            image_points,
            normalized_to_pixel(source["intrinsics"], width, height),
        )
        rotation = pose.pop("rotation")
        translation = pose.pop("translation")
        rvec, _ = cv2.Rodrigues(rotation.astype(np.float64))
        rows.append(
            {
                "source": old["source"],
                "target": old["target"],
                "scene": old["scene"],
                "matches": match_stats["ratio_matches"],
                "correspondences": match_stats["valid_correspondences"],
                "ok": True,
                "inliers": pose["inliers"],
                "inlier_ratio": pose["inlier_ratio"],
                "median_reprojection_px": pose["median_reprojection_px"],
                "p95_reprojection_px": pose["p95_reprojection_px"],
                "rvec": rvec.reshape(3).tolist(),
                "tvec": translation.tolist(),
                "intrinsics_mode": "source_frame_normalized_K_converted_to_pixel_K",
                "opencv_rng_seed": 7,
                "previous_pose": {
                    "rvec": old["rvec"],
                    "tvec": old["tvec"],
                    "median_reprojection_px": old["median_reprojection_px"],
                },
                **dense_displacement(
                    source["points"],
                    source["mask"],
                    source["intrinsics"],
                    rotation,
                    translation,
                ),
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "stage": "deterministic source-K pose re-estimation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "source_pairs": str(args.input_pairs.resolve()),
        "pair_count": len(rows),
        "pairs": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pair_count": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
