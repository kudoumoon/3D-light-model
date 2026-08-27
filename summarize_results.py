"""Rebuild audited summaries and figures from the bundled, pre-existing records.

No GPU, model weights, target-pose inference, or internet connection is needed.
See docs/AUDIT.md for the corrections relative to the original August 16 notes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latency_model import interval_cost


ROOT = Path(__file__).resolve().parent
SCENES = [("GTA driving", "gta_0000_bench", "gta_0000"),
          ("Temple Run", "temple_0000_bench", "temple"),
          ("Universal street", "universal_0003_bench", "universal")]
FIELDS = {"Copy": "copy_safe_tile_fraction_of_full",
          "3D warp": "safe_tile_fraction_of_full",
          "Tile-choice oracle": "tile_candidate_oracle_pass_fraction",
          "Pixel-min oracle (legacy)": "oracle_best_copy_or_warp_safe_tile_fraction_of_full"}
COLORS = {"Copy": "#777777", "3D warp": "#3473B8", "Tile-choice oracle": "#C39528",
          "Pixel-min oracle (legacy)": "#CB78A5"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def summarize_pairs(rows):
    return {"pairs": len(rows),
            "coverage_mean": mean(r["coverage_ratio"] for r in rows),
            "copy_psnr_coverage_mean": mean(r["copy_psnr_on_warp_coverage"] for r in rows),
            "warp_psnr_coverage_mean": mean(r["warp_psnr_on_coverage"] for r in rows),
            "warp_better_pixel_fraction_mean": mean(r["warp_better_than_copy_fraction_on_coverage"] for r in rows),
            "tile_pass_fraction": {label: {str(t): mean(r[field][str(t)] for r in rows)
                                         for t in [10, 20, 30, 40]}
                                   for label, field in FIELDS.items()}}


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)


def savefig(fig, path):
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "results/recorded")
    parser.add_argument("--pairs", type=Path, default=ROOT / "results/audited/quality_pairs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/audited")
    parser.add_argument("--figures", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    recorded = args.records
    checked = []
    geometry_rows = []
    for name, geo, warp in SCENES:
        g = read(recorded / geo / "metadata.json")
        w = read(recorded / f"{warp}_yaw5_cuda/reprojection_cuda.json")
        for label, samples, claimed in [(geo, g["inference_ms_all"], g["inference_ms_median"]),
                                         (warp, w["resident_gpu_ms_all"], w["resident_gpu_ms_median"])]:
            assert abs(median(samples) - claimed) <= .002, label
            checked.append(label)
        geometry_rows.append({"scene": name, "resolution": [g["width"], g["height"]],
                              "geometry_ms_median": g["inference_ms_median"],
                              "geometry_ms_mean": g["inference_ms_mean"],
                              "valid_ratio": g["valid_ratio"],
                              "peak_allocated_vram_mb": g["peak_allocated_vram_mb"],
                              "warp_ms_median": w["resident_gpu_ms_median"],
                              "upload_plus_warp_ms_median": w["upload_plus_gpu_ms_median"],
                              "coverage_yaw5": w["coverage_ratio"],
                              "coverage_by_yaw": {str(yaw): read(recorded / f"{warp}_yaw{yaw}/reprojection.json")["coverage_ratio"]
                                                  for yaw in [2, 5, 10]}})
    large = read(recorded / "active_dit_bench_large/active_dit_benchmark.json")
    small = read(recorded / "active_dit_bench/active_dit_benchmark.json")
    for label, data in [("large", large), ("small", small)]:
        for r in data["rows"]:
            assert abs(median(r["times_ms"]) - r["median_ms"]) <= .002
            checked.append(f"{label}:{r['active_ratio']}")
    cpu = read(recorded / "gta_0000_yaw5/reprojection.json")
    assert abs(median(cpu["reprojection_times_ms"]) - cpu["reprojection_ms_median"]) <= .002
    cpu_gpu = {"cpu_numpy_ms_median": cpu["reprojection_ms_median"],
               "cuda_resident_ms_median": geometry_rows[0]["warp_ms_median"],
               "h2d_cuda_ms_median": geometry_rows[0]["upload_plus_warp_ms_median"]}
    cpu_gpu["kernel_speedup"] = cpu_gpu["cpu_numpy_ms_median"] / cpu_gpu["cuda_resident_ms_median"]
    pairs = read(args.pairs)
    unique = {(r["track"], r["source_frame"], r["target_frame"]) for r in pairs["rows"]}
    assert len(unique) == len(pairs["rows"]), "Duplicate source-target records"
    quality = {}
    for scope, predicate in [("all_recorded", lambda r: True),
                             ("exclude_transition_risk", lambda r: not r["layout_transition_risk"])]:
        quality[scope] = {}
        for track in ["straight", "turning"]:
            selected = [r for r in pairs["rows"] if r["track"] == track and predicate(r)]
            quality[scope][track] = {group: summarize_pairs([r for r in selected if r["group"] == group])
                                     for group in ["short", "gap8", "long"]}
    speed = []
    for track in ["straight", "turning"]:
        rates = quality["all_recorded"][track]["gap8"]["tile_pass_fraction"]
        for method, thresholds in rates.items():
            for policy in ["floor", "legacy_linear"]:
                geo = geometry_rows[0]["geometry_ms_median"] if method != "Copy" else 0
                warp = geometry_rows[0]["warp_ms_median"] if method != "Copy" else 0
                speed.append({"track": track, "method": method, "policy": policy,
                              "f0_tile_pass_fraction": thresholds["20"],
                              **interval_cost(large["rows"], thresholds["20"], geo, warp, 8, policy)})
    report = {"experiment_date": "2026-08-16", "audit_date": "2026-08-28",
              "scope": "Component measurements and offline diagnostics; no real AR-DiT end-to-end speedup measured",
              "geometry": geometry_rows, "cpu_cuda": cpu_gpu,
              "quality": quality, "valid_pairs": len(pairs["rows"]), "failed_pairs": len(pairs["failures"]),
              "transition_risk_pairs": sum(r["layout_transition_risk"] for r in pairs["rows"]),
              "speed_ledger": {"warning": "Accounting only: target-dependent routes, proxy DiT, gap-8 rates used at every non-anchor age, tile/token mapping assumed, VAE/router/KV-build/display excluded. Floor is a sensitivity assumption, not a proven bound.",
                               "interval": 8, "threshold_rgb_mae_255": 20, "rows": speed},
              "validation": {"timing_medians_checked": checked, "unique_pairs": True}}
    write(args.output / "summary.json", report)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    labels = ["CPU / NumPy", "CUDA resident", "H2D + CUDA"]
    values = [cpu_gpu[k] for k in ["cpu_numpy_ms_median", "cuda_resident_ms_median", "h2d_cuda_ms_median"]]
    bars = ax.barh(labels, values, color="#3473B8", edgecolor="#274461")
    for bar, v in zip(bars, values):
        ax.text(v + 3, bar.get_y() + bar.get_height() / 2, f"{v:.3f} ms", va="center")
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel("Median component wall time (ms; zero-based linear axis)")
    ax.set_title("GTA point reprojection: CPU and CUDA\n640 x 326; yaw 5 deg + forward 0.1; radius 1; Aug 16, 2026")
    ax.invert_yaxis()
    style(ax)
    savefig(fig, args.figures / "warp_runtime.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, label, data in zip(axes, ["Large proxy", "Small proxy"], [large, small]):
        rows = data["rows"]
        ax.plot([100 * r["active_ratio"] for r in rows], [r["median_ms"] for r in rows],
                "o-", color="#3473B8", label="Measured median")
        full = next(r["median_ms"] for r in rows if r["active_ratio"] == 1)
        ax.plot([100 * r["active_ratio"] for r in rows], [full * r["active_ratio"] for r in rows],
                "--", color="#777777", label="Ideal linear reference")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Active query tokens (%)")
        ax.set_ylabel("One proxy evaluation (ms)")
        ax.set_title(f"{label}: {data['layers']} blocks\n{data['tokens']} tokens, width {data['width']}; resident K/V")
        style(ax)
        ax.legend(fontsize=8)
    fig.suptitle("Active-query microbenchmarks (not Matrix-Game); Aug 16, 2026")
    savefig(fig, args.figures / "active_query_scaling.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, constrained_layout=True)
    x = np.arange(2)
    for ax, scope, title in zip(axes, quality, ["All recorded: 13 pairs / track", "Before frame 90: 12 pairs / track"]):
        for offset, method, key in [(-.18, "Copy", "copy_psnr_coverage_mean"), (.18, "3D warp", "warp_psnr_coverage_mean")]:
            vals = [quality[scope][t]["short"][key] for t in ["straight", "turning"]]
            ax.bar(x + offset, vals, .34, label=method, color=COLORS[method], edgecolor="#444444")
            for px, v in zip(x + offset, vals):
                ax.text(px, v + .4, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, ["Straight / W", "Turning / W+A"])
        ax.set_title(title + "\nTwo correlated demo crops, gaps 3/6 frames")
        ax.set_ylim(0, 35)
        style(ax)
        ax.legend()
    axes[0].set_ylabel("Mean per-pair PSNR on warp coverage (dB)")
    fig.suptitle("Copy vs 3D warp: short-gap quality and montage-transition sensitivity")
    savefig(fig, args.figures / "copy_warp_quality.png")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, method in enumerate(FIELDS):
        vals = [100 * quality["all_recorded"][t]["gap8"]["tile_pass_fraction"][method]["20"] for t in ["straight", "turning"]]
        pos = x + (i - 1.5) * .2
        ax.bar(pos, vals, .19, label=method, color=COLORS[method], edgecolor="#444444", hatch="//" if i == 3 else None)
        for px, v in zip(pos, vals):
            ax.text(px, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["Straight / W", "Turning / W+A"])
    ax.set_ylabel("Tiles passing an RGB-MAE test (%)")
    ax.set_ylim(0, 123)
    ax.set_title("Gap-8 offline tile diagnostics\n16 x 16 tiles; MAE <= 20/255; >=95% warp support; six pairs per crop")
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    style(ax)
    savefig(fig, args.figures / "tile_oracle_audit.png")

    lines = ["# Audited result tables", "", "Generated by `python summarize_results.py`; timings are Aug 16 records, not a new run.", "", "## Short-gap quality", "", "| Scope | Crop | Pairs | Copy PSNR (dB) | Warp PSNR (dB) |", "|---|---|---:|---:|---:|"]
    for scope in quality:
        for track in ["straight", "turning"]:
            q = quality[scope][track]["short"]
            lines.append(f"| {scope} | {track} | {q['pairs']} | {q['copy_psnr_coverage_mean']:.3f} | {q['warp_psnr_coverage_mean']:.3f} |")
    lines += ["", "## Gap-8 tile pass rates", "", "All values are RGB threshold diagnostics, not validated safety probabilities.", "", "| Crop | Copy | Warp | Tile-choice oracle | Legacy pixel-min oracle |", "|---|---:|---:|---:|---:|"]
    for track in ["straight", "turning"]:
        q = quality["all_recorded"][track]["gap8"]["tile_pass_fraction"]
        lines.append("| " + track + " | " + " | ".join(f"{100*q[m]['20']:.3f}%" for m in FIELDS) + " |")
    lines += ["", "## Illustrative K=8 accounting — NOT measured system latency", "", report["speed_ledger"]["warning"], "", "| Crop | Route | Low-ratio policy | Average proxy ms | Proxy ratio |", "|---|---|---|---:|---:|"]
    for r in speed:
        lines.append(f"| {r['track']} | {r['method']} | {r['policy']} | {r['average_ms_proxy']:.3f} | {r['speedup_proxy']:.3f}x |")
    (args.output / "TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pairs": report["valid_pairs"], "failures": report["failed_pairs"],
                      "transition_risk_pairs": report["transition_risk_pairs"], "short_quality": quality["exclude_transition_risk"],
                      "floor_speed_ledger": [r for r in speed if r["policy"] == "floor"]}, indent=2))


if __name__ == "__main__":
    main()
