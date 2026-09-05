#!/usr/bin/env python3
"""从 Bonn 传感器 GT 位姿构建不同时间间隔的测试 pair，不读取或修改模型。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_bonn_latent3d_cache import build_pairs
from tools.run_tum_gt_latent3d_feasibility import synchronize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, action="append", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--pair-deltas", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--max-time-offset", type=float, default=0.03)
    args = parser.parse_args()
    if len(args.sequence_root) != len(args.scene):
        raise ValueError("--sequence-root 与 --scene 数量必须一致")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rows, sequence_stats = [], []
    for root, scene in zip(args.sequence_root, args.scene):
        synchronized = synchronize(root, args.max_time_offset)
        records = synchronized[:: args.frame_stride]
        sample_ids = [f"bonn_{scene}_{index:06d}" for index in range(len(records))]
        scene_rows = build_pairs(scene, records, sample_ids, args.pair_deltas, args.pair_stride)
        rows.extend(scene_rows)
        sequence_stats.append(
            {"scene": scene, "sampled_frames": len(records), "pairs": len(scene_rows)}
        )

    args.output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config": {
            "frame_stride": args.frame_stride,
            "pair_deltas_on_sampled_frames": args.pair_deltas,
            "pair_stride": args.pair_stride,
            "pose_source": "Bonn RGB-D sensor groundtruth",
        },
        "sequences": sequence_stats,
        "pairs": rows,
    }
    (args.output / "pairs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(rows), "sequences": sequence_stats}, indent=2))
    print("实验已完成")


if __name__ == "__main__":
    main()
