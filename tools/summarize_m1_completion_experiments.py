"""Summarize M1 completion experiments for cross-domain, hard-case, and downstream evidence.

This script does not train a model. It consolidates already recorded evidence into
paper-facing JSON/CSV/figures and exposes missing experiments explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def avg(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))]
    if not vals:
        raise KeyError(f"no finite values for {key}")
    return mean(vals)


def minv(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))]
    if not vals:
        raise KeyError(f"no finite values for {key}")
    return min(vals)


def maxv(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))]
    if not vals:
        raise KeyError(f"no finite values for {key}")
    return max(vals)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty csv rows: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_cross_domain(root: Path) -> dict:
    eval_dirs = [root / "results/recorded/reprojection_student_demo384_v2/eval_best"]
    eval_dirs.extend(sorted(root.glob("runs/reprojection_student/m1_selected_crossdomain_*_eval")))
    by_domain = []
    by_scene_csv = []
    missing = []
    for eval_dir in eval_dirs:
        summary = load_json(eval_dir / "summary.json")
        per_sample = load_jsonl(eval_dir / "per_sample.jsonl")
        domain = eval_dir.name.replace("m1_selected_crossdomain_", "").replace("_eval", "")
        if eval_dir.name == "eval_best":
            domain = "matrixgame2_demo_recorded"
        scene_names = sorted({row["scene"] for row in per_sample})
        scene_rows = []
        for scene in scene_names:
            rows = [row for row in per_sample if row["scene"] == scene]
            item = {
                "domain": domain,
                "scene": scene,
                "num_samples": len(rows),
                "loss": avg(rows, "loss"),
                "point": avg(rows, "point"),
                "projection": avg(rows, "projection"),
                "inference_ms_median": avg(rows, "inference_ms_median"),
            }
            scene_rows.append(item)
            by_scene_csv.append(item)
        by_domain.append({
            "domain": domain,
            "source": eval_dir.as_posix(),
            "num_samples": int(summary["num_samples"]),
            "loss": float(summary["loss"]),
            "point": float(summary["point"]),
            "projection": float(summary["projection"]),
            "inference_ms_median_mean": float(summary["inference_ms_median_mean"]),
            "by_scene": scene_rows,
        })
    total = sum(row["num_samples"] for row in by_domain)
    if total < 90:
        missing.append("cross-domain validation has fewer than 90 total samples; target is >=30 per domain for paper-grade evidence")
    return {
        "num_domains": len(by_domain),
        "num_samples_total": total,
        "by_domain": by_domain,
        "by_scene": by_scene_csv,
        "missing_for_paper": missing,
    }


def summarize_hard_cases(root: Path) -> dict:
    reproj = load_json(root / "results/recorded/m1_v10_frozen_geometry_motion_confidence/v10_reproj_summary.json")
    rows = reproj["rows"]
    by_scene_rows = []
    for scene, row in sorted(reproj["by_scene"].items(), key=lambda kv: kv[1]["coverage_gap_mean"]):
        by_scene_rows.append({
            "scene": scene,
            "num_cases": int(row["num_cases"]),
            "student_coverage_mean": float(row["student_coverage_mean"]),
            "teacher_coverage_mean": float(row["teacher_coverage_mean"]),
            "coverage_gap_mean": float(row["coverage_gap_mean"]),
            "coverage_gap_min": float(row["coverage_gap_min"]),
        })
    worst_cases = reproj.get("worst_cases", [])[:12]
    return {
        "source": "results/recorded/m1_v10_frozen_geometry_motion_confidence/v10_reproj_summary.json",
        "num_cases": int(reproj["num_cases"]),
        "coverage_gap_mean": float(reproj["coverage_gap_mean"]),
        "coverage_gap_min": float(reproj["coverage_gap_min"]),
        "yaw10_coverage_gap_mean": float(reproj["yaw10_coverage_gap_mean"]),
        "worst_scene": reproj["worst_scene"],
        "by_scene_ranked": by_scene_rows,
        "worst_cases": worst_cases,
        "repair_target": {
            "primary": "reduce coverage_gap_min from -0.200 toward >= -0.120 without increasing point/projection loss",
            "secondary": "improve game2_mid_left/game3_left coverage while preserving v10 confidence AUC/ECE",
        },
    }


def summarize_target_file(path: Path) -> dict:
    data = load_json(path)
    rows = data["rows"]
    motion_groups = {"low_motion": [], "high_motion": []}
    for row in rows:
        angle = abs(float(row.get("rotation_angle_deg", 0.0)))
        trans = float(row.get("translation_norm_predicted_metric", 0.0))
        group = "high_motion" if angle >= 0.5 or trans >= 0.05 else "low_motion"
        motion_groups[group].append(row)
    result = {
        "path": path.as_posix(),
        "num_pairs": len(rows),
        "failures": len(data.get("failures", [])),
        "pnp_inlier_ratio_mean": avg(rows, "pnp_inlier_ratio"),
        "pnp_reprojection_px_median_mean": avg(rows, "pnp_reprojection_px_median"),
        "coverage_ratio_mean": avg(rows, "coverage_ratio"),
        "warp_better_than_copy_fraction_mean": avg(rows, "warp_better_than_copy_fraction_on_coverage"),
        "copy_psnr_full_mean": avg(rows, "copy_psnr_full"),
        "warp_psnr_on_coverage_mean": avg(rows, "warp_psnr_on_coverage"),
    }
    for group, group_rows in motion_groups.items():
        if group_rows:
            result[group] = {
                "num_pairs": len(group_rows),
                "copy_psnr_full_mean": avg(group_rows, "copy_psnr_full"),
                "warp_psnr_on_coverage_mean": avg(group_rows, "warp_psnr_on_coverage"),
                "warp_better_than_copy_fraction_mean": avg(group_rows, "warp_better_than_copy_fraction_on_coverage"),
                "safe_tile_fraction_20_mean": mean(float(row["safe_tile_fraction_of_full"]["20"]) for row in group_rows),
                "copy_safe_tile_fraction_20_mean": mean(float(row["copy_safe_tile_fraction_of_full"]["20"]) for row in group_rows),
                "oracle_safe_tile_fraction_20_mean": mean(float(row["oracle_best_copy_or_warp_safe_tile_fraction_of_full"]["20"]) for row in group_rows),
            }
    return result


def summarize_hardcase_candidates(root: Path) -> dict:
    pattern = "runs/reprojection_student/student_video384_tvod_v11*_hardcoverage_*_reprojection_multi/summary.json"
    rows = []
    for path in sorted(root.glob(pattern)):
        data = load_json(path)
        run_name = path.parent.name.replace("_reprojection_multi", "")
        eval_path = root / "runs/reprojection_student" / f"{run_name}_eval" / "summary.json"
        eval_data = load_json(eval_path)
        rows.append({
            "run_name": run_name,
            "reprojection_summary": path.as_posix(),
            "eval_summary": eval_path.as_posix(),
            "num_cases": int(data["num_cases"]),
            "coverage_gap_mean": float(data["coverage_gap_mean"]),
            "coverage_gap_min": float(data["coverage_gap_min"]),
            "yaw10_coverage_gap_mean": float(data["yaw10_coverage_gap_mean"]),
            "point": float(eval_data["point"]),
            "projection": float(eval_data["projection"]),
            "inference_ms_median_mean": float(eval_data["inference_ms_median_mean"]),
        })
    selected = None
    if rows:
        selected = sorted(rows, key=lambda row: (row["coverage_gap_min"], row["coverage_gap_mean"], -row["projection"]), reverse=True)[0]
    return {
        "selection_rule": "maximize worst-case coverage gap first, then mean coverage gap, while monitoring projection loss",
        "num_candidates": len(rows),
        "selected_candidate": selected,
        "rows": rows,
    }


def summarize_downstream(root: Path) -> dict:
    files = [
        root / "results/recorded/matrix_game2_gta_target_eval/summary.json",
        root / "results/recorded/matrix_game2_gta_target_eval_k8/summary.json",
        root / "results/recorded/matrix_game2_gta_target_eval_long/summary.json",
        root / "results/recorded/matrix_game2_gta_turn_target_eval/summary.json",
        root / "results/recorded/matrix_game2_gta_turn_target_eval_k8/summary.json",
        root / "results/recorded/matrix_game2_gta_turn_target_eval_long/summary.json",
    ]
    target_summaries = [summarize_target_file(path) for path in files]
    large_bench = load_json(root / "results/recorded/active_dit_bench_large/active_dit_benchmark.json")
    active_rows = large_bench["rows"]
    speed_by_ratio = {float(row["active_ratio"]): float(row["speedup_vs_full"]) for row in active_rows}
    candidate_ratios = sorted(speed_by_ratio)
    def nearest_speedup(active_ratio: float) -> float:
        nearest = min(candidate_ratios, key=lambda ratio: abs(ratio - active_ratio))
        return speed_by_ratio[nearest]
    closed_loop_rows = []
    for item in target_summaries:
        high = item.get("high_motion")
        if high:
            reusable = high["safe_tile_fraction_20_mean"]
            active_ratio = max(0.125, min(1.0, 1.0 - reusable))
            closed_loop_rows.append({
                "source": item["path"],
                "split": "high_motion",
                "num_pairs": high["num_pairs"],
                "safe_tile_fraction_20": reusable,
                "estimated_active_ratio": active_ratio,
                "nearest_active_dit_speedup": nearest_speedup(active_ratio),
                "warp_better_than_copy_fraction": high["warp_better_than_copy_fraction_mean"],
            })
    return {
        "target_pair_summaries": target_summaries,
        "active_dit_benchmark_large": {
            "device": large_bench["device"],
            "tokens": large_bench["tokens"],
            "rows": [{"active_ratio": float(row["active_ratio"]), "median_ms": float(row["median_ms"]), "speedup_vs_full": float(row["speedup_vs_full"])} for row in active_rows],
        },
        "closed_loop_proxy": closed_loop_rows,
        "interpretation": "Use confidence/warp-safe tiles to skip reusable regions and route only unsafe regions to the expensive DiT path; this is a proxy until integrated end-to-end with the downstream renderer.",
    }



def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_bar_svg(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float], str]], y_label: str) -> None:
    width, height = 980, 460
    left, right, top, bottom = 80, 30, 60, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [value for _, vals, _ in series for value in vals]
    y_min = min(0.0, min(values))
    y_max = max(values)
    span = max(1e-6, y_max - y_min)
    n = len(labels)
    group_w = plot_w / max(1, n)
    bar_w = group_w / (len(series) + 1)
    def y(value: float) -> float:
        return top + (y_max - value) / span * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{y(0.0)}" x2="{left+plot_w}" y2="{y(0.0)}" stroke="#333"/>',
        f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" font-family="Arial" font-size="14">{svg_escape(y_label)}</text>',
    ]
    for i in range(5):
        value = y_min + span * i / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')
    for i, label in enumerate(labels):
        cx = left + i * group_w + group_w / 2
        parts.append(f'<text x="{cx:.1f}" y="{height-55}" text-anchor="end" transform="rotate(-30 {cx:.1f} {height-55})" font-family="Arial" font-size="12">{svg_escape(label)}</text>')
        for j, (name, vals, color) in enumerate(series):
            value = vals[i]
            x = left + i * group_w + (j + 0.5) * bar_w
            yy = y(max(value, 0.0)) if value >= 0 else y(0.0)
            hh = abs(y(value) - y(0.0))
            parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w*0.82:.1f}" height="{hh:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w*0.41:.1f}" y="{y(value)-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3f}</text>')
    lx = left + plot_w - 220
    for j, (name, _, color) in enumerate(series):
        parts.append(f'<rect x="{lx}" y="{top + j*22}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx+20}" y="{top + 12 + j*22}" font-family="Arial" font-size="13">{svg_escape(name)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_cross_domain(out_dir: Path, cross: dict) -> None:
    rows = cross["by_scene"]
    write_bar_svg(
        out_dir / "m1_cross_domain_demo_validation.svg",
        "M1 cross-domain demo validation",
        [row["scene"] for row in rows],
        [
            ("point loss", [row["point"] for row in rows], "#4c78a8"),
            ("projection loss", [row["projection"] for row in rows], "#f58518"),
        ],
        "loss",
    )


def plot_hard_cases(out_dir: Path, hard: dict) -> None:
    rows = hard["by_scene_ranked"]
    write_bar_svg(
        out_dir / "m1_hardcase_coverage_gap.svg",
        "M1 hard-case coverage gap by scene",
        [row["scene"] for row in rows],
        [("student - teacher coverage", [row["coverage_gap_mean"] for row in rows], "#c0392b")],
        "coverage gap",
    )


def plot_downstream(out_dir: Path, downstream: dict) -> None:
    rows = downstream["closed_loop_proxy"]
    write_bar_svg(
        out_dir / "m1_closed_loop_proxy_speedup.svg",
        "Closed-loop proxy: safe reprojection tiles reduce active DiT tokens",
        [Path(row["source"]).parent.name for row in rows],
        [
            ("estimated DiT speedup", [row["nearest_active_dit_speedup"] for row in rows], "#2980b9"),
            ("active token ratio", [row["estimated_active_ratio"] for row in rows], "#e67e22"),
        ],
        "speedup / ratio",
    )

def write_markdown(out_dir: Path, pack: dict) -> None:
    hard = pack["hard_cases"]
    down = pack["downstream"]
    best_proxy = max(down["closed_loop_proxy"], key=lambda row: row["nearest_active_dit_speedup"])
    lines = [
        "# M1 Completion Experiment Pack",
        "",
        "## 当前结论",
        "",
        f"- v10 当前 hard-case 平均 coverage gap 为 {hard['coverage_gap_mean']:.4f}，最差为 {hard['coverage_gap_min']:.4f}；主要问题场景是 `{hard['worst_scene']}`。",
        f"- 跨域验证当前覆盖 {pack['cross_domain']['num_domains']} 个域、{pack['cross_domain']['num_samples_total']} 个样本；论文级目标是每域 >=30。",
        f"- 下游闭环 proxy 显示，在高运动片段中可用安全重投影 tile 降低 active DiT token；当前最佳估计 speedup 为 {best_proxy['nearest_active_dit_speedup']:.2f}x。",
        f"- v11 hard-case 候选数：{pack['hardcase_candidates']['num_candidates']}；若候选训练完成，将按最差 coverage gap 优先自动排序。",
        "",
        "## 必须补齐的实验",
        "",
        "1. Cross-domain：GTA/Temple/Universal 每域至少 30 帧，导出 MoGe-3 teacher，再评估 M1-v7/v10 student。",
        "2. Hard-case repair：基于 v7 初始化，开启 coverage-deficit loss 与 depth-edge point loss，重点优化 game2_mid_left、game3_left 等 coverage gap 场景。",
        "3. Downstream closed-loop：用 real target frames 评估 copy/warp/oracle/gated-DiT 分层收益，按低运动/高运动分别报告。",
        "",
        "## 论文图输出",
        "",
        "- `figures/m1_cross_domain_demo_validation.svg`",
        "- `figures/m1_hardcase_coverage_gap.svg`",
        "- `figures/m1_closed_loop_proxy_speedup.svg`",
        "",
    ]
    (out_dir / "M1_COMPLETION_EXPERIMENT_PACK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/recorded/m1_v11_completion_pack")
    args = parser.parse_args()
    out_dir = args.output
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pack = {
        "cross_domain": summarize_cross_domain(ROOT),
        "hard_cases": summarize_hard_cases(ROOT),
        "hardcase_candidates": summarize_hardcase_candidates(ROOT),
        "downstream": summarize_downstream(ROOT),
    }
    (out_dir / "m1_completion_summary.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out_dir / "cross_domain_by_scene.csv", pack["cross_domain"]["by_scene"])
    write_csv(out_dir / "hardcase_by_scene.csv", pack["hard_cases"]["by_scene_ranked"])
    write_csv(out_dir / "closed_loop_proxy.csv", pack["downstream"]["closed_loop_proxy"])
    if pack["hardcase_candidates"]["rows"]:
        write_csv(out_dir / "hardcase_candidate_comparison.csv", pack["hardcase_candidates"]["rows"])
    plot_cross_domain(fig_dir, pack["cross_domain"])
    plot_hard_cases(fig_dir, pack["hard_cases"])
    plot_downstream(fig_dir, pack["downstream"])
    write_markdown(out_dir, pack)
    print(json.dumps({"output": out_dir.as_posix(), "figures": sorted(path.name for path in fig_dir.glob("*.svg"))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
