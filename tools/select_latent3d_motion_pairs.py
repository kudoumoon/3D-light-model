#!/usr/bin/env python3
"""Select a preregistered reliable-motion subset from a pose screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-inliers", type=int, default=200)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--max-median-reprojection-px", type=float, default=1.5)
    parser.add_argument("--min-median-displacement-px", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text())
    selected = [
        row
        for row in source["pairs"]
        if row.get("ok")
        and row["inliers"] >= args.min_inliers
        and row["inlier_ratio"] >= args.min_inlier_ratio
        and row["median_reprojection_px"] <= args.max_median_reprojection_px
        and row["dense_projected_displacement_px_median"]
        >= args.min_median_displacement_px
    ]
    report = {
        "schema_version": 1,
        "stage": "reliable measurable-motion subset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input.resolve()),
        "selection": {
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
            "max_median_reprojection_px": args.max_median_reprojection_px,
            "min_median_displacement_px": args.min_median_displacement_px,
            "selected_count": len(selected),
        },
        "pairs": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
