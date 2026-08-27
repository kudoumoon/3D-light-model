"""Compute a transparent break-even budget from measured local microbenchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "speed_gain"

SCENES = {
    "GTA driving": {
        "geometry": RESULTS / "gta_0000_bench" / "metadata.json",
        "warp": RESULTS / "gta_0000_yaw5_cuda" / "reprojection_cuda.json",
    },
    "Temple Run": {
        "geometry": RESULTS / "temple_0000_bench" / "metadata.json",
        "warp": RESULTS / "temple_yaw5_cuda" / "reprojection_cuda.json",
    },
    "Universal street": {
        "geometry": RESULTS / "universal_0003_bench" / "metadata.json",
        "warp": RESULTS / "universal_yaw5_cuda" / "reprojection_cuda.json",
    },
}

ROUTES = {
    # The only 4-step pixels are geometric holes. This is an optimistic oracle
    # bound because covered pixels have not yet been checked against target RGB.
    "holes_only_oracle": {"covered_to_2step": 0.0, "covered_to_4step": 0.0},
    # A transparent stress assumption, not a measured quality route: 20% of
    # covered pixels need 2 steps and another 10% need all 4 steps.
    "moderate_risk": {"covered_to_2step": 0.20, "covered_to_4step": 0.10},
    # Half of geometrically covered pixels still require neural compute.
    "high_risk": {"covered_to_2step": 0.25, "covered_to_4step": 0.25},
}


def make_latency_interpolator(rows: list[dict]):
    ratios = np.array([0.0] + [row["active_ratio"] for row in reversed(rows)], dtype=float)
    latency = np.array([0.0] + [row["median_ms"] for row in reversed(rows)], dtype=float)

    def lookup(ratio: float) -> float:
        return float(np.interp(np.clip(ratio, 0, 1), ratios, latency))

    return lookup


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dit = json.loads(
        (RESULTS / "active_dit_bench_large" / "active_dit_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    latency = make_latency_interpolator(dit["rows"])
    full_eval_ms = latency(1.0)
    full_four_step_ms = 4.0 * full_eval_ms
    intervals = [1, 4, 8, 16]
    rows = []

    for scene, paths in SCENES.items():
        geometry = json.loads(paths["geometry"].read_text(encoding="utf-8"))
        warp = json.loads(paths["warp"].read_text(encoding="utf-8"))
        coverage = float(warp["coverage_ratio"])
        geometry_ms = float(geometry["inference_ms_median"])
        warp_ms = float(warp["resident_gpu_ms_median"])
        for route_name, route in ROUTES.items():
            f2 = coverage * route["covered_to_2step"]
            f4 = (1.0 - coverage) + coverage * route["covered_to_4step"]
            f0 = 1.0 - f2 - f4
            active_step_1_2 = f2 + f4
            active_step_3_4 = f4
            routed_dit_ms = (
                2.0 * latency(active_step_1_2)
                + 2.0 * latency(active_step_3_4)
            )
            approximate_ms = warp_ms + routed_dit_ms
            for interval in intervals:
                if interval == 1:
                    # Every frame is exact, then refresh geometry. This is a
                    # control rather than an acceleration configuration.
                    average_ms = full_four_step_ms + geometry_ms
                else:
                    average_ms = (
                        full_four_step_ms
                        + geometry_ms
                        + (interval - 1) * approximate_ms
                    ) / interval
                rows.append(
                    {
                        "scene": scene,
                        "route": route_name,
                        "exact_interval": interval,
                        "coverage": round(coverage, 6),
                        "f0": round(f0, 6),
                        "f2": round(f2, 6),
                        "f4": round(f4, 6),
                        "geometry_ms": geometry_ms,
                        "warp_ms": warp_ms,
                        "routed_dit_ms_non_exact": round(routed_dit_ms, 3),
                        "average_ms": round(average_ms, 3),
                        "speedup": round(full_four_step_ms / average_ms, 3),
                        "unmeasured_overhead_margin_ms": round(
                            full_four_step_ms - average_ms, 3
                        ),
                        "positive_before_unmeasured_overheads": average_ms < full_four_step_ms,
                    }
                )

    report = {
        "scope": "Measured local geometry/warp plus measured DiT-like proxy; not Matrix-Game end-to-end",
        "baseline": {
            "one_full_dit_like_eval_ms": round(full_eval_ms, 3),
            "four_step_full_ms": round(full_four_step_ms, 3),
            "proxy": {
                key: dit[key]
                for key in ["device", "dtype", "tokens", "width", "heads", "layers", "mlp_ratio"]
            },
        },
        "accounting": (
            "One exact four-step frame and one geometry refresh per interval; "
            "all other frames pay CUDA warp plus packed active-query DiT. "
            "VAE/router/verification are not measured and must fit inside the reported margin."
        ),
        "rows": rows,
    }
    (OUT / "speed_gain.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, constrained_layout=True)
    colors = {"holes_only_oracle": "#4daf4a", "moderate_risk": "#377eb8", "high_risk": "#e41a1c"}
    labels = {"holes_only_oracle": "Holes only (oracle)", "moderate_risk": "Moderate risk", "high_risk": "High risk"}
    for axis, scene in zip(axes, SCENES):
        for route_name in ROUTES:
            selected = [
                row for row in rows
                if row["scene"] == scene and row["route"] == route_name and row["exact_interval"] > 1
            ]
            axis.plot(
                [row["exact_interval"] for row in selected],
                [row["speedup"] for row in selected],
                marker="o", linewidth=2, color=colors[route_name], label=labels[route_name],
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(scene)
        axis.set_xlabel("Exact/geometry refresh interval")
        axis.set_xticks([4, 8, 16])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Speedup vs full four-step proxy")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Break-even model from measured local kernels (unmeasured overhead excluded)")
    fig.savefig(OUT / "speed_gain_by_scene.png", dpi=200)
    plt.close(fig)

    cpu_warp = json.loads(
        (RESULTS / "gta_0000_yaw5" / "reprojection.json").read_text(encoding="utf-8")
    )
    cuda_warp = json.loads(
        (RESULTS / "gta_0000_yaw5_cuda" / "reprojection_cuda.json").read_text(encoding="utf-8")
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    names = ["CPU / NumPy", "CUDA resident", "H2D + CUDA"]
    values = [
        cpu_warp["reprojection_ms_median"],
        cuda_warp["resident_gpu_ms_median"],
        cuda_warp["upload_plus_gpu_ms_median"],
    ]
    bars = axis.bar(names, values, color=["#e41a1c", "#4daf4a", "#377eb8"])
    axis.set_yscale("log")
    axis.set_ylim(top=max(values) * 1.45)
    axis.set_ylabel("Median latency (ms, log scale)")
    axis.set_title("GTA 640x326, 5 deg yaw, radius-1 z-buffer splat")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.2f} ms", ha="center")
    fig.savefig(OUT / "warp_cpu_vs_cuda.png", dpi=200)
    plt.close(fig)

    focus = [
        row for row in rows
        if row["scene"] == "GTA driving" and row["exact_interval"] == 8
    ]
    print(json.dumps({"baseline": report["baseline"], "gta_interval_8": focus}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
