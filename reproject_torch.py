"""CUDA z-buffer point splatting for the GameWarp geometry baseline.

The timed resident path assumes geometry is already on the GPU, which matches the
intended streaming system.  A second benchmark includes CPU-to-GPU upload so the
transfer cost is not hidden.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from reproject import _rotation_c2w


@torch.inference_mode()
def forward_splat_torch(
    points_source: torch.Tensor,
    rgb_source: torch.Tensor,
    mask_source: torch.Tensor,
    intrinsics_normalized: torch.Tensor,
    camera_center_source: torch.Tensor,
    target_c2source_rotation: torch.Tensor,
    splat_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a resident point map using CUDA scatter-reduce z buffering."""
    height, width = mask_source.shape
    valid = mask_source & torch.isfinite(points_source).all(dim=-1)
    xyz_source = points_source[valid]
    colors = rgb_source[valid]

    xyz_target = (xyz_source - camera_center_source) @ target_c2source_rotation
    z = xyz_target[:, 2]
    front = torch.isfinite(z) & (z > 1e-4)
    xyz_target, colors, z = xyz_target[front], colors[front], z[front]

    xy = xyz_target[:, :2] / z[:, None]
    k = intrinsics_normalized
    u_norm = k[0, 0] * xy[:, 0] + k[0, 1] * xy[:, 1] + k[0, 2]
    v_norm = k[1, 0] * xy[:, 0] + k[1, 1] * xy[:, 1] + k[1, 2]
    u0 = torch.round(u_norm * width - 0.5).to(torch.int64)
    v0 = torch.round(v_norm * height - 0.5).to(torch.int64)

    offsets = torch.arange(-splat_radius, splat_radius + 1, device=z.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dx, dy = dx.flatten(), dy.flatten()
    u = u0[:, None] + dx[None, :]
    v = v0[:, None] + dy[None, :]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    flat = (v * width + u)[inside]
    candidate_depth = z[:, None].expand_as(u)[inside]
    color_index = torch.arange(len(z), device=z.device, dtype=torch.int64)
    color_index = color_index[:, None].expand_as(u)[inside]

    pixel_count = height * width
    target_depth = torch.full(
        (pixel_count,), torch.inf, dtype=torch.float32, device=z.device
    )
    target_depth.scatter_reduce_(
        0, flat, candidate_depth, reduce="amin", include_self=True
    )

    # Recover one deterministic source-color index among candidates at the
    # nearest depth.  This avoids a GPU sort and keeps the operation O(N).
    nearest = torch.isclose(
        candidate_depth, target_depth[flat], rtol=1e-5, atol=1e-5
    )
    sentinel = len(z)
    candidate_color = torch.where(
        nearest, color_index, torch.full_like(color_index, sentinel)
    )
    winner = torch.full(
        (pixel_count,), sentinel, dtype=torch.int64, device=z.device
    )
    winner.scatter_reduce_(
        0, flat, candidate_color, reduce="amin", include_self=True
    )
    target_mask = winner < sentinel
    warped = torch.zeros((pixel_count, 3), dtype=torch.uint8, device=z.device)
    warped[target_mask] = colors[winner[target_mask]]
    target_depth = torch.where(target_mask, target_depth, 0.0)
    return (
        warped.reshape(height, width, 3),
        target_depth.reshape(height, width),
        target_mask.reshape(height, width),
    )


def _time_cuda(call, warmup: int, repeat: int) -> tuple[tuple, list[float]]:
    for _ in range(max(0, warmup)):
        result = call()
    torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result, times


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA point reprojection")
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--right", type=float, default=0.0)
    parser.add_argument("--down", type=float, default=0.0)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    data = np.load(args.geometry, allow_pickle=False)
    cpu_arrays = {
        "points": torch.from_numpy(data["points"]),
        "rgb": torch.from_numpy(data["rgb"]),
        "mask": torch.from_numpy(data["mask"]),
        "intrinsics": torch.from_numpy(data["intrinsics"]),
    }
    resident = {name: value.to(device) for name, value in cpu_arrays.items()}
    center = torch.tensor(
        [args.right, args.down, args.forward], dtype=torch.float32, device=device
    )
    rotation = torch.from_numpy(_rotation_c2w(args.yaw, args.pitch)).to(device)
    radius = max(0, args.splat_radius)

    def resident_call():
        return forward_splat_torch(
            resident["points"], resident["rgb"], resident["mask"],
            resident["intrinsics"], center, rotation, radius,
        )

    result, resident_times = _time_cuda(resident_call, args.warmup, args.repeat)

    def upload_call():
        uploaded = {name: value.to(device) for name, value in cpu_arrays.items()}
        return forward_splat_torch(
            uploaded["points"], uploaded["rgb"], uploaded["mask"],
            uploaded["intrinsics"], center, rotation, radius,
        )

    _, upload_times = _time_cuda(upload_call, args.warmup, args.repeat)
    warped, depth, mask = (value.cpu().numpy() for value in result)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(args.output / "warped_rgb.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(str(args.output / "warped_mask.png"), mask.astype(np.uint8) * 255)
    np.save(args.output / "warped_depth.npy", depth)
    overlay = warped.copy()
    overlay[~mask] = np.array([255, 0, 255], dtype=np.uint8)
    cv2.imwrite(
        str(args.output / "warped_holes_magenta.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    report = {
        "device": torch.cuda.get_device_name(),
        "geometry": str(args.geometry.resolve()),
        "resolution": [int(mask.shape[1]), int(mask.shape[0])],
        "yaw_deg": args.yaw,
        "translation_right_down_forward": [args.right, args.down, args.forward],
        "splat_radius": radius,
        "coverage_ratio": round(float(mask.mean()), 6),
        "resident_gpu_ms_all": [round(v, 3) for v in resident_times],
        "resident_gpu_ms_mean": round(float(np.mean(resident_times)), 3),
        "resident_gpu_ms_median": round(float(np.median(resident_times)), 3),
        "upload_plus_gpu_ms_all": [round(v, 3) for v in upload_times],
        "upload_plus_gpu_ms_mean": round(float(np.mean(upload_times)), 3),
        "upload_plus_gpu_ms_median": round(float(np.median(upload_times)), 3),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    (args.output / "reprojection_cuda.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
