"""Create group-meeting figures from the local geometry baseline results."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "report_assets"

SCENES = {
    "GTA driving": {
        "geometry": RESULTS / "gta_0000_bench",
        "warp": {2: RESULTS / "gta_0000_yaw2", 5: RESULTS / "gta_0000_yaw5", 10: RESULTS / "gta_0000_yaw10"},
        "slug": "gta",
    },
    "Temple Run": {
        "geometry": RESULTS / "temple_0000_bench",
        "warp": {2: RESULTS / "temple_yaw2", 5: RESULTS / "temple_yaw5", 10: RESULTS / "temple_yaw10"},
        "slug": "temple",
    },
    "Universal street": {
        "geometry": RESULTS / "universal_0003_bench",
        "warp": {2: RESULTS / "universal_yaw2", 5: RESULTS / "universal_yaw5", 10: RESULTS / "universal_yaw10"},
        "slug": "universal",
    },
}


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_montage(name: str, item: dict) -> None:
    root = item["geometry"]
    images = [
        (read_rgb(root / "rgb.png"), "Input RGB"),
        (read_rgb(root / "depth_vis.png"), "MoGe-2 metric depth"),
        (read_rgb(root / "normal_vis.png"), "Predicted normals"),
        (read_rgb(item["warp"][5] / "warped_holes_magenta.png"), "5 deg yaw warp (magenta = holes)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (image, title) in zip(axes.ravel(), images):
        axis.imshow(image)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    fig.suptitle(name, fontsize=16, fontweight="bold")
    fig.savefig(OUT / f"{item['slug']}_geometry_montage.png", dpi=180)
    plt.close(fig)


def make_pointcloud(name: str, item: dict) -> None:
    data = np.load(item["geometry"] / "geometry.npz", allow_pickle=False)
    points, colors, mask = data["points"], data["rgb"], data["mask"]
    valid = mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    # Single-view predictors can assign extremely large depths to sky/far
    # background.  A robust 90th-percentile crop keeps the local geometry
    # legible without changing the saved point cloud used by reprojection.
    z_values = points[..., 2][valid]
    z_limit = min(np.percentile(z_values, 90), np.median(z_values) * 6.0)
    sample = np.zeros_like(valid)
    sample[::3, ::3] = True
    valid &= sample & (points[..., 2] <= z_limit)
    xyz = points[valid]
    rgb = colors[valid].astype(np.float32) / 255
    x_low, x_high = np.percentile(xyz[:, 0], [1, 99])
    up_low, up_high = np.percentile(-xyz[:, 1], [1, 99])

    fig = plt.figure(figsize=(10, 7))
    axis = fig.add_subplot(111, projection="3d")
    # Plot x-right, z-forward, -y-up for an intuitive scene view.
    axis.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], c=rgb, s=0.35, linewidths=0)
    axis.set_xlabel("x / right")
    axis.set_ylabel("z / forward")
    axis.set_zlabel("up")
    axis.set_title(f"{name}: feed-forward metric point cloud")
    axis.view_init(elev=22, azim=-72)
    axis.set_box_aspect((1.6, 2.2, 1.0))
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(0, z_limit)
    axis.set_zlim(up_low, up_high)
    fig.savefig(OUT / f"{item['slug']}_pointcloud.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, item in SCENES.items():
        make_montage(name, item)
        make_pointcloud(name, item)
        metadata = json.loads((item["geometry"] / "metadata.json").read_text(encoding="utf-8"))
        coverage = {
            yaw: json.loads((path / "reprojection.json").read_text(encoding="utf-8"))["coverage_ratio"]
            for yaw, path in item["warp"].items()
        }
        summary[name] = {
            "resolution": f"{metadata['width']}x{metadata['height']}",
            "inference_ms_mean": metadata["inference_ms_mean"],
            "inference_ms_median": metadata["inference_ms_median"],
            "peak_allocated_vram_mb": metadata["peak_allocated_vram_mb"],
            "geometry_valid_ratio": metadata["valid_ratio"],
            "coverage": coverage,
        }

    yaws = [2, 5, 10]
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for name, values in summary.items():
        axis.plot(yaws, [100 * values["coverage"][yaw] for yaw in yaws], marker="o", linewidth=2, label=name)
    axis.set_xlabel("Target camera yaw (degrees)")
    axis.set_ylabel("Forward-warp coverage (%)")
    axis.set_xticks(yaws)
    axis.set_ylim(35, 100)
    axis.grid(alpha=0.3)
    axis.legend()
    axis.set_title("Reprojection coverage falls with camera motion and invalid geometry")
    fig.savefig(OUT / "coverage_vs_yaw.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    names = list(summary)
    axes[0].bar(names, [summary[name]["inference_ms_mean"] for name in names], color="#377eb8")
    axes[0].set_ylabel("Steady inference (ms)")
    axes[0].set_ylim(0, 75)
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, [100 * summary[name]["geometry_valid_ratio"] for name in names], color="#4daf4a")
    axes[1].set_ylabel("Valid geometry pixels (%)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("MoGe-2-ViT-S on RTX 4060 Laptop GPU")
    fig.savefig(OUT / "runtime_and_validity.png", dpi=200)
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
