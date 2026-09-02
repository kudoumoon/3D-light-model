#!/usr/bin/env python3
"""Aggregate P13/P14 upgrade experiments without changing raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def relative_delta(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/latent3d"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    p13_names = (
        "p13_reprojection_control_seed7_v1",
        "p13_reprojection_l1_seed7_v1",
        "p13_reprojection_l1cos_seed7_v1",
    )
    p13 = {name: load(args.root / name) for name in p13_names}
    p13_rows = [
        {
            "experiment": name,
            "best_epoch": report["selection"]["best_epoch"],
            "validation_warp_l1": report["selection"]["best_validation_warp_l1"],
            "test_warp_l1": report["test_pairs"]["all"]["warp_l1"],
            "hard_motion_warp_l1": report["test_pairs"]["hard_motion"]["warp_l1"],
            "test_abs_rel": report["test_geometry"]["abs_rel"],
        }
        for name, report in p13.items()
    ]

    p14_paths = {
        7: ("p14_virtual_projective_evalonly_seed7_v1", "p14_virtual_projective_feature_coord_seed7_v1"),
        11: ("p14_virtual_projective_evalonly_seed11_v1", "p14_virtual_projective_feature_coord_seed11_v1"),
        23: ("p14_virtual_projective_feature_coord_seed23_v1", "p14_virtual_projective_feature_coord_seed23_v1"),
    }
    metric_paths = {
        "virtual_feature_l1": ("test_virtual", "feature_l1"),
        "virtual_coordinate_l1": ("test_virtual", "coordinate_l1"),
        "virtual_coverage_gap": ("test_virtual", "coverage_gap"),
        "geometry_abs_rel": ("test_geometry", "abs_rel"),
        "geometry_projection_l1": ("test_geometry", "projection_l1"),
        "geometry_valid_iou": ("test_geometry", "valid_iou"),
        "pair_warp_l1": ("test_estimated_pose_pairs", "all", "warp_l1"),
        "hard_motion_warp_l1": ("test_estimated_pose_pairs", "hard_motion", "warp_l1"),
    }

    def value(report: dict, path: tuple[str, ...]) -> float:
        current = report
        for key in path:
            current = current[key]
        return float(current)

    p14_rows = []
    for seed, (baseline_name, candidate_name) in p14_paths.items():
        baseline = load(args.root / baseline_name)
        candidate = load(args.root / candidate_name)
        row: dict[str, object] = {
            "seed": seed,
            "baseline": baseline_name,
            "candidate": candidate_name,
            "best_epoch": candidate["selection"]["best_epoch"],
        }
        for metric, path in metric_paths.items():
            baseline_value = value(baseline, path)
            candidate_value = value(candidate, path)
            row[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "relative_delta": relative_delta(candidate_value, baseline_value),
            }
        p14_rows.append(row)

    aggregates = {}
    for metric in metric_paths:
        deltas = np.asarray([row[metric]["relative_delta"] for row in p14_rows])
        lower_is_better = metric != "geometry_valid_iou"
        wins = deltas < 0 if lower_is_better else deltas > 0
        aggregates[metric] = {
            "mean_relative_delta": float(deltas.mean()),
            "std_relative_delta": float(deltas.std()),
            "wins": int(wins.sum()),
            "seeds": len(deltas),
        }

    report = {
        "schema_version": 1,
        "p13_direct_target_latent_reprojection": {
            "rows": p13_rows,
            "finding": "All variants selected epoch 0; direct estimated-pose target-latent fine-tuning produced no validated gain.",
            "status": "Negative Result",
        },
        "p14_virtual_projective_distillation": {
            "rows": p14_rows,
            "aggregate": aggregates,
            "finding": "One seed improved, one was mixed, and one reverted to initialization; the upgrade is not cross-seed stable.",
            "status": "Hypothesis",
        },
        "decision": "Keep the existing 64x3 latent geometry head as the deliverable baseline. Do not promote P13/P14 until fresh GT-pose multi-view validation succeeds.",
        "evidence_boundary": "Matrix-Game generated video frames; pair poses are MoGe-assisted PnP estimates; test scenes were inspected during method development.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
