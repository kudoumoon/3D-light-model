#!/usr/bin/env python3
"""Audit whether a latent-warp screen contains meaningful camera motion."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-downsample", type=float, default=8.0)
    return parser.parse_args()


def key(source: str, target: str) -> tuple[str, str]:
    return source, target


def summarize(rows: list[dict]) -> dict:
    l1 = [
        float(row["warp_latent_l1_valid"])
        - float(row["copy_latent_l1_same_valid"])
        for row in rows
    ]
    psnr = [
        float(row["decoded_composite_psnr"])
        - float(row["decoded_copy_psnr"])
        for row in rows
    ]
    return {
        "pair_count": len(rows),
        "valid_l1_delta_mean": statistics.fmean(l1),
        "valid_l1_win_rate": sum(value < 0 for value in l1) / len(l1),
        "composite_psnr_delta_mean": statistics.fmean(psnr),
        "composite_psnr_delta_median": statistics.median(psnr),
        "composite_psnr_win_rate": sum(value > 0 for value in psnr) / len(psnr),
    }


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text())
    pose_manifest = json.loads(args.pairs.read_text())
    displacement = {
        key(row["source"], row["target"]): float(
            row["dense_projected_displacement_px_median"]
        )
        for row in pose_manifest["pairs"]
    }
    joined = []
    missing = []
    for row in metrics["rows"]:
        pair = key(row["pair"]["source"], row["pair"]["target"])
        if pair not in displacement:
            missing.append(pair)
            continue
        joined.append({**row, "median_projected_displacement_px": displacement[pair]})
    bins = (
        ("subpixel_lt1", 0.0, 1.0),
        ("small_1to4", 1.0, 4.0),
        ("meaningful_ge4", 4.0, float("inf")),
    )
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        by_method[row["alignment"]].append(row)
    aggregates = {}
    for method, rows in sorted(by_method.items()):
        method_bins = {}
        for label, lower, upper in bins:
            selected = [
                row
                for row in rows
                if lower <= row["median_projected_displacement_px"] < upper
            ]
            if selected:
                method_bins[label] = {
                    **summarize(selected),
                    "median_projected_displacement_px_mean": statistics.fmean(
                        row["median_projected_displacement_px"] for row in selected
                    ),
                }
        aggregates[method] = {"all": summarize(rows), "motion_bins": method_bins}

    unique_displacements = sorted(
        {
            (row["pair"]["source"], row["median_projected_displacement_px"])
            for row in joined
        }
    )
    values = [value for _, value in unique_displacements]
    meaningful_count = sum(value >= 4.0 for value in values)
    latent_cell_count = sum(value >= args.spatial_downsample for value in values)
    report = {
        "schema_version": 1,
        "metrics": str(args.metrics),
        "pose_manifest": str(args.pairs),
        "spatial_downsample": args.spatial_downsample,
        "joined_row_count": len(joined),
        "missing_pair_rows": sorted(set(missing)),
        "unique_pair_count": len(values),
        "motion_sufficiency": {
            "median_projected_displacement_px_median": statistics.median(values),
            "median_projected_displacement_px_mean": statistics.fmean(values),
            "pairs_ge_1px": sum(value >= 1.0 for value in values),
            "pairs_ge_4px": meaningful_count,
            "pairs_ge_one_latent_cell": latent_cell_count,
            "sufficient_for_camera_motion_claim": meaningful_count >= 20,
            "minimum_requested_meaningful_pairs": 20,
        },
        "evidence_status": (
            "motion_sufficient"
            if meaningful_count >= 20
            else "motion_insufficient_do_not_generalize_to_camera_motion"
        ),
        "aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"evidence_status": report["evidence_status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
