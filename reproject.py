"""Forward-warp a MoGe geometry frame to a user-specified target camera."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _rotation_c2w(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Target camera-to-source rotation in OpenCV coordinates.

    Positive yaw turns right. Positive pitch looks down. Coordinates are
    x-right, y-down, z-forward.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float32,
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(-pitch), -np.sin(-pitch)],
         [0, np.sin(-pitch), np.cos(-pitch)]],
        dtype=np.float32,
    )
    return ry @ rx


def forward_splat(
    points_source: np.ndarray,
    rgb_source: np.ndarray,
    mask_source: np.ndarray,
    intrinsics_normalized: np.ndarray,
    camera_center_source: np.ndarray,
    target_c2source_rotation: np.ndarray,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask_source.shape
    valid = mask_source & np.isfinite(points_source).all(axis=-1)
    xyz_source = points_source[valid]
    colors = rgb_source[valid]
    # X_target = R_c2s^T (X_source - C_source)
    xyz_target = (target_c2source_rotation.T @ (xyz_source - camera_center_source).T).T
    z = xyz_target[:, 2]
    front = np.isfinite(z) & (z > 1e-4)
    xyz_target, colors, z = xyz_target[front], colors[front], z[front]

    x = xyz_target[:, 0] / z
    y = xyz_target[:, 1] / z
    k = intrinsics_normalized
    u_norm = k[0, 0] * x + k[0, 1] * y + k[0, 2]
    v_norm = k[1, 0] * x + k[1, 1] * y + k[1, 2]
    u0 = np.rint(u_norm * width - 0.5).astype(np.int32)
    v0 = np.rint(v_norm * height - 0.5).astype(np.int32)

    offsets = [(dy, dx) for dy in range(-splat_radius, splat_radius + 1)
               for dx in range(-splat_radius, splat_radius + 1)]
    all_u, all_v, all_z, all_c = [], [], [], []
    for dy, dx in offsets:
        u, v = u0 + dx, v0 + dy
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        all_u.append(u[inside])
        all_v.append(v[inside])
        all_z.append(z[inside])
        all_c.append(colors[inside])
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    z = np.concatenate(all_z)
    colors = np.concatenate(all_c)
    flat = v.astype(np.int64) * width + u
    # Sort by pixel, then nearest depth. First item per pixel wins.
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    selected = order[first]

    warped = np.zeros((height * width, 3), dtype=np.uint8)
    target_depth = np.full(height * width, np.inf, dtype=np.float32)
    warped[flat[selected]] = colors[selected]
    target_depth[flat[selected]] = z[selected]
    target_mask = np.isfinite(target_depth)
    target_depth[~target_mask] = 0
    return (
        warped.reshape(height, width, 3),
        target_depth.reshape(height, width),
        target_mask.reshape(height, width),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward reproject an extracted geometry frame")
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--right", type=float, default=0.0,
                        help="Target camera translation to its right, in predicted metric units")
    parser.add_argument("--down", type=float, default=0.0)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="Positive turns camera right")
    parser.add_argument("--pitch", type=float, default=0.0, help="Positive looks down")
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Untimed in-process reprojection runs")
    parser.add_argument("--repeat", type=int, default=5,
                        help="Timed in-process reprojection runs")
    args = parser.parse_args()

    data = np.load(args.geometry, allow_pickle=False)
    camera_center = np.array([args.right, args.down, args.forward], dtype=np.float32)
    rotation = _rotation_c2w(args.yaw, args.pitch)
    call_args = (
        data["points"], data["rgb"], data["mask"], data["intrinsics"],
        camera_center, rotation, max(0, args.splat_radius),
    )
    for _ in range(max(0, args.warmup)):
        forward_splat(*call_args)
    times_ms = []
    result = None
    for _ in range(max(1, args.repeat)):
        start = time.perf_counter()
        result = forward_splat(*call_args)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    warped, depth, mask = result
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "warped_rgb.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output / "warped_mask.png"), mask.astype(np.uint8) * 255)
    np.save(args.output / "warped_depth.npy", depth)
    overlay = warped.copy()
    overlay[~mask] = np.array([255, 0, 255], dtype=np.uint8)
    cv2.imwrite(str(args.output / "warped_holes_magenta.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    report = {
        "geometry": str(args.geometry.resolve()),
        "camera_translation_right_down_forward": camera_center.tolist(),
        "yaw_deg": args.yaw,
        "pitch_deg": args.pitch,
        "splat_radius": args.splat_radius,
        "coverage_ratio": round(float(mask.mean()), 6),
        "hole_ratio": round(float(1.0 - mask.mean()), 6),
        "reprojection_device": "CPU / NumPy",
        "reprojection_times_ms": [round(value, 3) for value in times_ms],
        "reprojection_ms_mean": round(float(np.mean(times_ms)), 3),
        "reprojection_ms_median": round(float(np.median(times_ms)), 3),
    }
    (args.output / "reprojection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
