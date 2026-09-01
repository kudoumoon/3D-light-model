#!/usr/bin/env python3
"""Build a deterministic pseudo-pose screen with measurable camera motion."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.reestimate_latent3d_pairs import dense_displacement
from tools.run_rgb_intrinsics_pose_ablation import (
    collect_correspondences,
    estimate_pose,
    normalized_to_pixel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-gaps", type=int, nargs="+", default=(12, 24, 48))
    parser.add_argument("--pairs-per-scene-gap", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def frame_number(sample_id: str) -> int:
    match = re.search(r"frame_(\d+)$", sample_id)
    if match is None:
        raise ValueError(f"sample id has no frame number: {sample_id}")
    return int(match.group(1))


def select_tasks(records: list[dict], gaps: tuple[int, ...], count: int) -> list[dict]:
    by_scene: dict[str, dict[int, dict]] = {}
    for record in records:
        by_scene.setdefault(record["scene"], {})[frame_number(record["sample_id"])] = record
    tasks = []
    for scene, indexed in sorted(by_scene.items()):
        for gap in gaps:
            candidates = [
                (frame, indexed[frame], indexed[frame + gap])
                for frame in sorted(indexed)
                if frame + gap in indexed
            ]
            if not candidates:
                continue
            selected_indices = np.linspace(
                0, len(candidates) - 1, min(count, len(candidates)), dtype=int
            )
            for index in selected_indices:
                frame, source, target = candidates[index]
                tasks.append(
                    {
                        "scene": scene,
                        "frame_gap": gap,
                        "source_frame": frame,
                        "source": source,
                        "target": target,
                    }
                )
    return tasks


def process_task(payload: tuple[str, dict]) -> dict:
    teacher_root_string, task = payload
    teacher_root = Path(teacher_root_string)
    source_record, target_record = task["source"], task["target"]
    base = {
        "source": source_record["sample_id"],
        "target": target_record["sample_id"],
        "scene": task["scene"],
        "frame_gap": task["frame_gap"],
        "source_frame": task["source_frame"],
        "intrinsics_mode": "source_frame_normalized_K_converted_to_pixel_K",
        "opencv_rng_seed": 7,
    }
    try:
        source = np.load(teacher_root / source_record["geometry"])
        target = np.load(teacher_root / target_record["geometry"])
        objects, images, match_stats = collect_correspondences(
            source["rgb"], target["rgb"], source["points"], source["mask"]
        )
        height, width = source["mask"].shape
        pose = estimate_pose(
            objects,
            images,
            normalized_to_pixel(source["intrinsics"], width, height),
        )
        rotation = pose.pop("rotation")
        translation = pose.pop("translation")
        rvec, _ = cv2.Rodrigues(rotation.astype(np.float64))
        return {
            **base,
            "ok": True,
            "matches": match_stats["ratio_matches"],
            "correspondences": match_stats["valid_correspondences"],
            "inliers": pose["inliers"],
            "inlier_ratio": pose["inlier_ratio"],
            "median_reprojection_px": pose["median_reprojection_px"],
            "p95_reprojection_px": pose["p95_reprojection_px"],
            "rvec": rvec.reshape(3).tolist(),
            "tvec": translation.tolist(),
            **dense_displacement(
                source["points"], source["mask"], source["intrinsics"], rotation, translation
            ),
        }
    except Exception as error:
        return {**base, "ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.pairs_per_scene_gap <= 0:
        raise ValueError("workers and pairs-per-scene-gap must be positive")
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    tasks = select_tasks(
        manifest["records"], tuple(args.frame_gaps), args.pairs_per_scene_gap
    )
    payloads = [(str(args.teacher_root.resolve()), task) for task in tasks]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(process_task, payloads))
    rows.sort(key=lambda row: (row["scene"], row["frame_gap"], row["source_frame"]))
    reliable = [
        row
        for row in rows
        if row.get("ok")
        and row["inliers"] >= 200
        and row["inlier_ratio"] >= 0.6
        and row["median_reprojection_px"] <= 1.5
    ]
    meaningful = [
        row for row in reliable if row["dense_projected_displacement_px_median"] >= 4.0
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "stage": "deterministic measurable-motion pseudo-pose screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "teacher_root": str(args.teacher_root.resolve()),
        "selection": {
            "frame_gaps": args.frame_gaps,
            "pairs_per_scene_gap": args.pairs_per_scene_gap,
            "workers": args.workers,
            "candidate_count": len(rows),
            "successful_pnp_count": sum(bool(row.get("ok")) for row in rows),
            "reliable_count": len(reliable),
            "meaningful_motion_count": len(meaningful),
            "reliable_gate": "inliers>=200, inlier_ratio>=0.6, median_reprojection<=1.5px",
            "motion_gate": "median dense projected displacement>=4px",
        },
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
