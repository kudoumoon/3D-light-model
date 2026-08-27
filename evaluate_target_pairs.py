"""Evaluate MoGe reprojection against real future frames from a demo trajectory.

The official demo video does not expose camera telemetry, so target-frame SIFT
matches plus source MoGe 3D points are used to recover a pseudo-ground-truth
relative pose with PnP.  This pose estimation is an offline evaluation oracle,
not a proposed runtime input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from reproject_torch import forward_splat_torch
from quality_metrics import tile_candidate_oracle_fraction


def normalized_to_pixel_intrinsics(k: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.diag([width, height, 1.0]).astype(np.float32)
    pixel = scale @ k.astype(np.float32)
    pixel[0, 2] -= 0.5
    pixel[1, 2] -= 0.5
    return pixel


def load_target(path: Path, width: int, height: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def estimate_pose_pnp(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    points: np.ndarray,
    valid_geometry: np.ndarray,
    intrinsics_pixel: np.ndarray,
) -> dict:
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    key_source, desc_source = sift.detectAndCompute(source_gray, None)
    key_target, desc_target = sift.detectAndCompute(target_gray, None)
    if desc_source is None or desc_target is None:
        raise RuntimeError("SIFT did not find descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_source, desc_target, k=2)
    matches = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]

    object_points, image_points, accepted_matches = [], [], []
    height, width = valid_geometry.shape
    for match in matches:
        u, v = key_source[match.queryIdx].pt
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < width and 0 <= y < height and valid_geometry[y, x]):
            continue
        point = points[y, x]
        if not np.isfinite(point).all() or point[2] <= 0:
            continue
        object_points.append(point)
        image_points.append(key_target[match.trainIdx].pt)
        accepted_matches.append(match)
    if len(object_points) < 12:
        raise RuntimeError(f"Only {len(object_points)} valid 3D-2D matches")

    object_points_np = np.asarray(object_points, dtype=np.float32)
    image_points_np = np.asarray(image_points, dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points_np,
        image_points_np,
        intrinsics_pixel,
        None,
        iterationsCount=500,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 8:
        raise RuntimeError("PnP RANSAC failed")
    indices = inliers.ravel()
    rvec_ransac, tvec_ransac = rvec.copy(), tvec.copy()
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        object_points_np[indices], image_points_np[indices],
        intrinsics_pixel, None, rvec, tvec,
    )
    projected_ransac, _ = cv2.projectPoints(
        object_points_np[indices], rvec_ransac, tvec_ransac, intrinsics_pixel, None
    )
    error_ransac = np.linalg.norm(
        projected_ransac[:, 0] - image_points_np[indices], axis=1
    )
    projected_refined, _ = cv2.projectPoints(
        object_points_np[indices], rvec_refined, tvec_refined, intrinsics_pixel, None
    )
    error_refined = np.linalg.norm(
        projected_refined[:, 0] - image_points_np[indices], axis=1
    )
    refine_accepted = bool(
        np.isfinite(error_refined).all()
        and np.median(error_refined) <= np.median(error_ransac) * 1.05
    )
    if refine_accepted:
        rvec, tvec, error = rvec_refined, tvec_refined, error_refined
    else:
        rvec, tvec, error = rvec_ransac, tvec_ransac, error_ransac
    if not np.isfinite(error).all() or np.median(error) > 5.0:
        raise RuntimeError(
            f"PnP pose rejected: median reprojection error {np.median(error):.2f}px"
        )
    rotation, _ = cv2.Rodrigues(rvec)
    angle_deg = math.degrees(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return {
        "rotation_source_to_target": rotation.astype(np.float32),
        "translation_source_to_target": tvec.reshape(3).astype(np.float32),
        "sift_source": len(key_source),
        "sift_target": len(key_target),
        "ratio_matches": len(matches),
        "valid_3d2d_matches": len(object_points),
        "pnp_inliers": int(len(indices)),
        "pnp_inlier_ratio": float(len(indices) / len(object_points)),
        "pnp_reprojection_px_median": float(np.median(error)),
        "pnp_reprojection_px_p95": float(np.percentile(error, 95)),
        "pnp_refine_lm_accepted": refine_accepted,
        "rotation_angle_deg": float(angle_deg),
        "translation_norm_predicted_metric": float(np.linalg.norm(tvec)),
    }


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    difference = (a[mask].astype(np.float32) - b[mask].astype(np.float32)) / 255.0
    mse = float(np.mean(difference ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def tile_safe_fraction(error: np.ndarray, mask: np.ndarray, threshold: float, tile: int) -> float:
    height, width = mask.shape
    safe, total = 0, 0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            tile_mask = mask[y:min(y + tile, height), x:min(x + tile, width)]
            tile_error = error[y:min(y + tile, height), x:min(x + tile, width)]
            total += 1
            if tile_mask.mean() >= 0.95 and tile_error[tile_mask].mean() <= threshold:
                safe += 1
    return safe / max(1, total)


def save_montage(
    output: Path,
    source: np.ndarray,
    target: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
    error: np.ndarray,
    title: str,
) -> None:
    holes = warped.copy()
    holes[~mask] = np.array([255, 0, 255], dtype=np.uint8)
    error_vis = cv2.applyColorMap(
        np.clip(error / 0.25 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    error_vis = cv2.cvtColor(error_vis, cv2.COLOR_BGR2RGB)
    error_vis[~mask] = 0
    images = [
        (source, "Source"), (target, "Real future target"),
        (holes, "3D warp (magenta holes)"), (error_vis, "Covered-pixel RGB MAE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), constrained_layout=True)
    for axis, (image, label) in zip(axes.ravel(), images):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.inference_mode()
def evaluate_pair(
    geometry_path: Path,
    target_path: Path,
    output: Path,
    source_frame: int,
    target_frame: int,
    tile: int,
) -> dict:
    data = np.load(geometry_path, allow_pickle=False)
    points, source_rgb = data["points"], data["rgb"]
    geometry_mask, k_norm = data["mask"], data["intrinsics"]
    height, width = geometry_mask.shape
    target_rgb = load_target(target_path, width, height)
    k_pixel = normalized_to_pixel_intrinsics(k_norm, width, height)
    pose = estimate_pose_pnp(
        source_rgb, target_rgb, points, geometry_mask, k_pixel
    )
    r_source_to_target = pose.pop("rotation_source_to_target")
    t_source_to_target = pose.pop("translation_source_to_target")
    rotation_c2s = r_source_to_target.T
    camera_center_source = -rotation_c2s @ t_source_to_target

    device = torch.device("cuda")
    result = forward_splat_torch(
        torch.from_numpy(points).to(device),
        torch.from_numpy(source_rgb).to(device),
        torch.from_numpy(geometry_mask).to(device),
        torch.from_numpy(k_norm).to(device),
        torch.from_numpy(camera_center_source).to(device),
        torch.from_numpy(rotation_c2s).to(device),
        splat_radius=1,
    )
    warped, _, warp_mask = (value.cpu().numpy() for value in result)
    pixel_error = np.mean(
        np.abs(warped.astype(np.float32) - target_rgb.astype(np.float32)), axis=-1
    ) / 255.0
    copy_error = np.mean(
        np.abs(source_rgb.astype(np.float32) - target_rgb.astype(np.float32)), axis=-1
    ) / 255.0
    oracle_best_error = np.minimum(
        copy_error, np.where(warp_mask, pixel_error, np.inf)
    )
    thresholds = [10 / 255, 20 / 255, 30 / 255, 40 / 255]
    report = {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "delta_frames": target_frame - source_frame,
        "resolution": [width, height],
        **pose,
        "camera_center_source": camera_center_source.tolist(),
        "target_c2source_rotation": rotation_c2s.tolist(),
        "pose_scope": "target-assisted offline PnP; not deployable action-to-pose",
        "coverage_ratio": float(warp_mask.mean()),
        "copy_psnr_full": psnr(source_rgb, target_rgb, np.ones_like(warp_mask)),
        "copy_psnr_on_warp_coverage": psnr(source_rgb, target_rgb, warp_mask),
        "warp_psnr_on_coverage": psnr(warped, target_rgb, warp_mask),
        "copy_mae_on_coverage": float(copy_error[warp_mask].mean()),
        "warp_mae_on_coverage": float(pixel_error[warp_mask].mean()),
        "safe_pixel_fraction_of_full": {
            str(round(value * 255)): float((warp_mask & (pixel_error <= value)).mean())
            for value in thresholds
        },
        "safe_pixel_fraction_of_covered": {
            str(round(value * 255)): float((pixel_error[warp_mask] <= value).mean())
            for value in thresholds
        },
        "copy_safe_pixel_fraction_of_full": {
            str(round(value * 255)): float((copy_error <= value).mean())
            for value in thresholds
        },
        "copy_safe_pixel_fraction_on_warp_coverage": {
            str(round(value * 255)): float((copy_error[warp_mask] <= value).mean())
            for value in thresholds
        },
        "oracle_best_copy_or_warp_safe_fraction_of_full": {
            str(round(value * 255)): float((oracle_best_error <= value).mean())
            for value in thresholds
        },
        "warp_better_than_copy_fraction_on_coverage": float(
            (pixel_error[warp_mask] < copy_error[warp_mask]).mean()
        ),
        "safe_tile_fraction_of_full": {
            str(round(value * 255)): tile_safe_fraction(pixel_error, warp_mask, value, tile)
            for value in thresholds
        },
        "copy_safe_tile_fraction_of_full": {
            str(round(value * 255)): tile_safe_fraction(
                copy_error, np.ones_like(warp_mask, dtype=bool), value, tile
            )
            for value in thresholds
        },
        "oracle_best_copy_or_warp_safe_tile_fraction_of_full": {
            str(round(value * 255)): tile_safe_fraction(
                oracle_best_error, np.ones_like(warp_mask, dtype=bool), value, tile
            )
            for value in thresholds
        },
        "legacy_oracle_definition": "per-pixel minimum copy/warp error, then tile aggregation; NOT a tile candidate selector",
        "tile_candidate_oracle_pass_fraction": {
            str(round(value * 255)): tile_candidate_oracle_fraction(
                copy_error, pixel_error, warp_mask, value, tile
            ) for value in thresholds
        },
        "tile_size": tile,
    }
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "source.png"), cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "target.png"), cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "warped.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "mask.png"), warp_mask.astype(np.uint8) * 255)
    np.save(output / "pixel_mae.npy", pixel_error)
    save_montage(
        output / "montage.png", source_rgb, target_rgb, warped, warp_mask,
        pixel_error, f"Matrix-Game 2 demo tile: {source_frame} -> {target_frame}",
    )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=int, nargs="+", required=True)
    parser.add_argument("--deltas", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--tile", type=int, default=16)
    args = parser.parse_args()

    if args.tile <= 0 or any(delta <= 0 for delta in args.deltas):
        raise ValueError("positive tile size and future-frame deltas required")
    cv2.setRNGSeed(7)
    rows, failures, skipped = [], [], []
    for source in args.sources:
        for delta in args.deltas:
            target = source + delta
            geometry_path = args.geometry / f"frame_{source:04d}" / "geometry.npz"
            target_path = args.frames / f"frame_{target:04d}.png"
            if not geometry_path.exists() or not target_path.exists():
                skipped.append({"source": source, "target": target, "reason": "missing geometry or target file"})
                continue
            pair_out = args.output / f"{source:04d}_to_{target:04d}"
            try:
                row = evaluate_pair(
                    geometry_path, target_path, pair_out, source, target, args.tile
                )
                rows.append(row)
                print(
                    f"{source:04d}->{target:04d}: coverage={row['coverage_ratio']:.3f}, "
                    f"warp_psnr={row['warp_psnr_on_coverage']:.2f}, "
                    f"safe20={row['safe_pixel_fraction_of_full']['20']:.3f}"
                )
            except Exception as error:  # preserve other pairs for audit
                failures.append({"source": source, "target": target, "error": repr(error)})
                print(f"FAILED {source:04d}->{target:04d}: {error}")
    summary = {"rows": rows, "failures": failures, "skipped": skipped, "opencv_rng_seed": 7}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"pairs": len(rows), "failures": failures}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
