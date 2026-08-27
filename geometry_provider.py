"""Unified RGB -> geometry provider for Matrix-Game-style experiments.

The first implementation wraps MoGe-2.  It intentionally exports a small,
model-agnostic NPZ contract so that MoGe can later be replaced by VGGT,
Depth Anything, engine G-buffers, or a jointly-trained AR-DiT geometry head.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_rgb(path: Path, max_size: int | None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max_size and max(rgb.shape[:2]) > max_size:
        scale = max_size / max(rgb.shape[:2])
        rgb = cv2.resize(
            rgb,
            (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return rgb


def _depth_visualization(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    lo, hi = np.percentile(depth[valid], [2, 98])
    normalized = 1.0 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    stride: int,
) -> int:
    sample = np.zeros_like(mask, dtype=bool)
    sample[::stride, ::stride] = True
    valid = mask & sample & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    xyz = points[valid].astype("<f4", copy=False)
    rgb = colors[valid].astype("u1", copy=False)
    records = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    records["x"], records["y"], records["z"] = xyz.T
    records["red"], records["green"], records["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)
    return len(records)


def run_moge(
    image_path: Path,
    output_dir: Path,
    model_name: str,
    max_size: int,
    num_tokens: int,
    fov_x: float | None,
    ply_stride: int,
    warmup: int,
    repeat: int,
) -> dict:
    third_party = _repo_root() / "third_party" / "MoGe"
    if not third_party.exists():
        raise FileNotFoundError(f"MoGe repository is missing: {third_party}")
    sys.path.insert(0, str(third_party))
    from moge.model.v2 import MoGeModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline")
    device = torch.device("cuda")
    rgb = _load_rgb(image_path, max_size)
    image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).to(device)

    load_start = time.perf_counter()
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    model_load_ms = (time.perf_counter() - load_start) * 1000
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            output = model.infer(
                image_tensor, fov_x=fov_x, num_tokens=num_tokens, use_fp16=True
            )
        timings_ms = []
        for _ in range(max(1, repeat)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.infer(
                image_tensor, fov_x=fov_x, num_tokens=num_tokens, use_fp16=True
            )
            torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000)

    points = output["points"].float().cpu().numpy()
    depth = output["depth"].float().cpu().numpy()
    mask = output["mask"].bool().cpu().numpy()
    intrinsics = output["intrinsics"].float().cpu().numpy()
    normal = output.get("normal")
    normal_np = normal.float().cpu().numpy() if normal is not None else np.empty((0,), np.float32)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth_vis = _depth_visualization(depth, mask)
    cv2.imwrite(str(output_dir / "depth_vis.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "mask.png"), mask.astype(np.uint8) * 255)
    if normal_np.size:
        normal_vis = np.clip((normal_np + 1) * 127.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "normal_vis.png"), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

    np.savez_compressed(
        output_dir / "geometry.npz",
        points=points.astype(np.float32),
        depth=depth.astype(np.float32),
        mask=mask,
        intrinsics=intrinsics.astype(np.float32),
        normal=normal_np.astype(np.float32),
        rgb=rgb,
        coordinate_convention=np.array("opencv_x_right_y_down_z_forward"),
    )
    num_vertices = _write_binary_ply(
        output_dir / "pointcloud.ply", points, rgb, mask, max(1, ply_stride)
    )
    valid_depth = depth[mask & np.isfinite(depth) & (depth > 0)]
    report = {
        "provider": "moge2",
        "model": model_name,
        "input": str(image_path.resolve()),
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
        "num_tokens": num_tokens,
        "model_load_ms": round(model_load_ms, 3),
        "warmup_runs": max(0, warmup),
        "timed_runs": max(1, repeat),
        "inference_ms_all": [round(value, 3) for value in timings_ms],
        "inference_ms_mean": round(float(np.mean(timings_ms)), 3),
        "inference_ms_median": round(float(np.median(timings_ms)), 3),
        "peak_allocated_vram_mb": round(peak_vram_mb, 1),
        "valid_ratio": round(float(mask.mean()), 6),
        "depth_median": round(float(np.median(valid_depth)), 5),
        "depth_p02": round(float(np.percentile(valid_depth, 2)), 5),
        "depth_p98": round(float(np.percentile(valid_depth, 98)), 5),
        "pointcloud_vertices": num_vertices,
        "intrinsics_normalized": intrinsics.tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract depth/points/intrinsics from an RGB frame")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--ply-stride", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    report = run_moge(
        args.image,
        args.output,
        args.model,
        args.max_size,
        args.num_tokens,
        args.fov_x,
        args.ply_stride,
        args.warmup,
        args.repeat,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
