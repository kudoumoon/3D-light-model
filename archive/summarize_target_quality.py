"""Aggregate real-future-frame quality and feed measured safe tiles into speed accounting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "target_quality_summary"

TRACKS = {
    "Straight / W": RESULTS / "matrix_game2_gta_target_eval" / "summary.json",
    "Turning / W+A": RESULTS / "matrix_game2_gta_turn_target_eval" / "summary.json",
}
K8_TRACKS = {
    "Straight / W": RESULTS / "matrix_game2_gta_target_eval_k8" / "summary.json",
    "Turning / W+A": RESULTS / "matrix_game2_gta_turn_target_eval_k8" / "summary.json",
}
LONG_TRACKS = {
    "Straight / W": RESULTS / "matrix_game2_gta_target_eval_long" / "summary.json",
    "Turning / W+A": RESULTS / "matrix_game2_gta_turn_target_eval_long" / "summary.json",
}
SAFE_FIELDS = {
    "Copy": "copy_safe_tile_fraction_of_full",
    "3D warp": "safe_tile_fraction_of_full",
    "Oracle best(copy, warp)": "oracle_best_copy_or_warp_safe_tile_fraction_of_full",
}


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def mean_threshold(rows: list[dict], field: str, threshold: int) -> float:
    return float(np.mean([row[field][str(threshold)] for row in rows]))


def latency_interpolator(rows: list[dict]):
    ratios = np.array([0.0] + [row["active_ratio"] for row in reversed(rows)])
    times = np.array([0.0] + [row["median_ms"] for row in reversed(rows)])
    return lambda ratio: float(np.interp(np.clip(ratio, 0, 1), ratios, times))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for track, path in TRACKS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        summaries[track] = {
            "pairs": len(rows),
            "failures": payload["failures"],
            "coverage_mean": mean(rows, "coverage_ratio"),
            "copy_psnr_coverage_mean": mean(rows, "copy_psnr_on_warp_coverage"),
            "warp_psnr_coverage_mean": mean(rows, "warp_psnr_on_coverage"),
            "warp_better_pixel_fraction_mean": mean(
                rows, "warp_better_than_copy_fraction_on_coverage"
            ),
            "rotation_angle_deg_mean": mean(rows, "rotation_angle_deg"),
            "pnp_inliers_mean": mean(rows, "pnp_inliers"),
            "pnp_reprojection_px_median_mean": mean(
                rows, "pnp_reprojection_px_median"
            ),
            "safe_tile_fraction": {
                label: {
                    str(threshold): mean_threshold(rows, field, threshold)
                    for threshold in [10, 20, 30, 40]
                }
                for label, field in SAFE_FIELDS.items()
            },
        }

    k8_summaries = {}
    for track, path in K8_TRACKS.items():
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        k8_summaries[track] = {
            "pairs": len(rows),
            "copy_psnr_coverage_mean": mean(rows, "copy_psnr_on_warp_coverage"),
            "warp_psnr_coverage_mean": mean(rows, "warp_psnr_on_coverage"),
            "warp_better_pixel_fraction_mean": mean(
                rows, "warp_better_than_copy_fraction_on_coverage"
            ),
            "safe_tile_fraction": {
                label: mean_threshold(rows, field, 20)
                for label, field in SAFE_FIELDS.items()
            },
        }

    thresholds = [10, 20, 30, 40]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True, constrained_layout=True)
    colors = {"Copy": "#777777", "3D warp": "#377eb8", "Oracle best(copy, warp)": "#4daf4a"}
    for axis, (track, values) in zip(axes, summaries.items()):
        for label in SAFE_FIELDS:
            axis.plot(
                thresholds,
                [100 * values["safe_tile_fraction"][label][str(v)] for v in thresholds],
                marker="o", linewidth=2, color=colors[label], label=label,
            )
        axis.set_title(track)
        axis.set_xlabel("Tile mean RGB-MAE threshold (/255)")
        axis.set_xticks(thresholds)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Safe 16x16 tiles (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Real future-frame route oracle on Matrix-Game 2 demo crops")
    fig.savefig(OUT / "safe_tiles_copy_warp_oracle.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    tracks = list(summaries)
    x = np.arange(len(tracks))
    width = 0.34
    copy_values = [summaries[t]["copy_psnr_coverage_mean"] for t in tracks]
    warp_values = [summaries[t]["warp_psnr_coverage_mean"] for t in tracks]
    axis.bar(x - width / 2, copy_values, width, label="Copy previous frame", color="#777777")
    axis.bar(x + width / 2, warp_values, width, label="3D warp", color="#377eb8")
    axis.set_xticks(x, tracks)
    axis.set_ylabel("PSNR on warp-covered pixels (dB)")
    axis.set_ylim(0, 34)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("3D warp is not automatically better than temporal copy")
    for position, value in zip(x - width / 2, copy_values):
        axis.text(position, value + 0.5, f"{value:.2f}", ha="center")
    for position, value in zip(x + width / 2, warp_values):
        axis.text(position, value + 0.5, f"{value:.2f}", ha="center")
    fig.savefig(OUT / "copy_vs_warp_psnr.png", dpi=200)
    plt.close(fig)

    # Combine 0.1/0.2 s, exact K=8 (0.267 s), and 0.5/1.0 s evaluations.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True, constrained_layout=True)
    temporal_summary = {}
    for axis, track in zip(axes, TRACKS):
        rows = []
        for mapping in [TRACKS, K8_TRACKS, LONG_TRACKS]:
            rows.extend(json.loads(mapping[track].read_text(encoding="utf-8"))["rows"])
        grouped = {}
        for delta in sorted({row["delta_frames"] for row in rows}):
            selected = [row for row in rows if row["delta_frames"] == delta]
            grouped[str(delta)] = {
                "pairs": len(selected),
                "time_sec": delta / 30.0,
                "copy_psnr": mean(selected, "copy_psnr_on_warp_coverage"),
                "warp_psnr": mean(selected, "warp_psnr_on_coverage"),
                "copy_safe_tile20": mean_threshold(
                    selected, "copy_safe_tile_fraction_of_full", 20
                ),
                "warp_safe_tile20": mean_threshold(
                    selected, "safe_tile_fraction_of_full", 20
                ),
                "best_safe_tile20": mean_threshold(
                    selected, "oracle_best_copy_or_warp_safe_tile_fraction_of_full", 20
                ),
            }
        temporal_summary[track] = grouped
        values = list(grouped.values())
        x_time = [value["time_sec"] for value in values]
        axis.plot(x_time, [value["copy_psnr"] for value in values], marker="o", linewidth=2,
                  color="#777777", label="Copy")
        axis.plot(x_time, [value["warp_psnr"] for value in values], marker="o", linewidth=2,
                  color="#377eb8", label="3D warp")
        axis.set_title(track)
        axis.set_xlabel("Future-frame gap (seconds)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("PSNR on warp-covered pixels (dB)")
    axes[1].legend()
    fig.suptitle("Quality stress test from 0.1 to 1.0 seconds")
    fig.savefig(OUT / "psnr_vs_temporal_gap.png", dpi=200)
    plt.close(fig)

    dit = json.loads(
        (RESULTS / "active_dit_bench_large" / "active_dit_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    t_dit = latency_interpolator(dit["rows"])
    full_four_step = 4 * t_dit(1.0)
    exact_interval = 8
    geometry_ms = 58.606
    warp_ms = 3.693
    speed_rows = []
    for track, values in k8_summaries.items():
        for method in SAFE_FIELDS:
            f0 = values["safe_tile_fraction"][method]
            f4 = 1.0 - f0
            approximate_dit = 4 * t_dit(f4)
            uses_geometry = method != "Copy"
            approximate_ms = approximate_dit + (warp_ms if uses_geometry else 0.0)
            average_ms = (
                full_four_step
                + (geometry_ms if uses_geometry else 0.0)
                + (exact_interval - 1) * approximate_ms
            ) / exact_interval
            speed_rows.append(
                {
                    "track": track,
                    "method": method,
                    "threshold_rgb_mae_255": 20,
                    "safe_tile_fraction_f0": round(f0, 6),
                    "regenerate_fraction_f4": round(f4, 6),
                    "average_ms_proxy": round(average_ms, 3),
                    "speedup_proxy": round(full_four_step / average_ms, 3),
                }
            )

    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    methods = list(SAFE_FIELDS)
    width = 0.24
    for offset, method in enumerate(methods):
        values = [
            next(row["speedup_proxy"] for row in speed_rows if row["track"] == track and row["method"] == method)
            for track in tracks
        ]
        positions = x + (offset - 1) * width
        axis.bar(positions, values, width, label=method, color=colors[method])
        for position, value in zip(positions, values):
            axis.text(position, value + 0.08, f"{value:.2f}x", ha="center", fontsize=9)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, tracks)
    axis.set_ylabel("Speedup vs full four-step DiT-like proxy")
    axis.set_ylim(0, max(row["speedup_proxy"] for row in speed_rows) * 1.22)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("K=8 measured-quality 0/4 oracle fed into the local speed ledger")
    fig.savefig(OUT / "quality_conditioned_speedup.png", dpi=200)
    plt.close(fig)

    report = {
        "scope": (
            "Official Matrix-Game 2 demo montage crops; target-assisted PnP and route "
            "selection are offline oracles, not runtime inputs."
        ),
        "pair_count": sum(value["pairs"] for value in summaries.values()),
        "summaries": summaries,
        "k8_summaries": k8_summaries,
        "temporal_summary": temporal_summary,
        "speed_accounting": {
            "exact_interval": exact_interval,
            "tile_threshold_rgb_mae_255": 20,
            "four_step_proxy_ms": round(full_four_step, 3),
            "geometry_ms": geometry_ms,
            "warp_ms": warp_ms,
            "rows": speed_rows,
        },
    }
    (OUT / "target_quality_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
