#!/usr/bin/env python3
"""构建真实 TUM RGB-D Student 训练清单。

几何监督只来自 TUM sensor depth；pose 和官方内参写入 provenance，训练脚本
随后会把 depth 对齐到 frozen VAE 的 44x80 latent grid。该工具不调用 MoGe。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_tum_gt_latent3d_feasibility import FR1_K, synchronize


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz-root", type=Path, required=True)
    parser.add_argument("--rpy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-per-sequence", type=int, default=240)
    return parser.parse_args()


def export_sequence(root: Path, output: Path, prefix: str, stride: int, limit: int) -> list[dict[str, str]]:
    records = synchronize(root, tolerance=0.03)[::stride][:limit]
    normalized_k = FR1_K.copy()
    normalized_k[0] /= 640.0
    normalized_k[1] /= 480.0
    exported = []
    for index, record in enumerate(records):
        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.uint8)
        raw_depth = np.asarray(Image.open(record["depth"]), dtype=np.uint16)
        depth = raw_depth.astype(np.float32) / 5000.0
        valid = ((depth > 0.2) & (depth < 8.0)).astype(np.float32)
        relative = Path(prefix) / f"{index:06d}"
        destination = output / relative
        destination.mkdir(parents=True, exist_ok=False)
        geometry_path = destination / "geometry.npz"
        np.savez_compressed(
            geometry_path,
            rgb=rgb,
            depth=depth,
            mask=valid,
            intrinsics=normalized_k,
        )
        exported.append(
            {
                "sample_id": f"{prefix}_{index:06d}",
                "scene": f"{prefix}_train" if index % 10 < 8 else f"{prefix}_val" if index % 10 == 8 else f"{prefix}_test",
                "geometry": str(relative / "geometry.npz"),
                "source_timestamp": str(record["timestamp"]),
            }
        )
    return exported


def main() -> None:
    args = arguments()
    if args.stride < 1 or args.max_per_sequence < 1:
        raise ValueError("stride and max-per-sequence must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    records = export_sequence(args.xyz_root, args.output, "tum_xyz", args.stride, args.max_per_sequence)
    rpy_records = export_sequence(args.rpy_root, args.output, "tum_rpy", args.stride, args.max_per_sequence)
    records.extend(rpy_records)
    manifest = {
        "dataset": "tum_rgbd_sensor_depth_mocap_pose",
        "supervision": "TUM registered sensor depth; no MoGe geometry",
        "records": records,
        "protocol": {
            "xyz_train": "scene used for training",
            "xyz_val": "validation only",
            "xyz_test": "held out static-motion subset",
            "rpy_test": "cross-sequence pure-rotation subset",
            "rgb_depth_pose_sync_tolerance_s": 0.03,
            "depth_scale": 5000.0,
            "intrinsics": "Freiburg-1 official pinhole K normalized by native 640x480 size",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
