#!/usr/bin/env python3
"""Audit latent3d experiment metrics before they are used as paper evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return statistics.fmean(values)


def bootstrap_mean_interval(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    generator = random.Random(seed)
    means = sorted(
        average([values[generator.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def pair_key(row: dict) -> str:
    pair = row["pair"]
    return f'{pair["scene"]}:{pair["source"]}->{pair["target"]}'


def validate_grain(report: dict) -> dict:
    rows = report["rows"]
    methods = report["methods"]
    keys = [(pair_key(row), row["alignment"]) for row in rows]
    unique_pairs = sorted({key[0] for key in keys})
    duplicate_count = len(keys) - len(set(keys))
    expected_rows = len(unique_pairs) * len(methods)
    return {
        "declared_pair_count": report["pair_count"],
        "observed_pair_count": len(unique_pairs),
        "method_count": len(methods),
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "duplicate_pair_method_rows": duplicate_count,
        "complete_factorial": len(rows) == expected_rows and duplicate_count == 0,
    }


def copy_consistency(report: dict) -> dict:
    if int(report["schema_version"]) == 2:
        # Same-valid metrics intentionally vary with each alignment method's
        # projected mask. Only full-frame Copy metrics must be invariant.
        fields = (
            ("copy_latent_l1_full", "copy_latent_l1_full"),
            ("decoded_copy_psnr", "decoded_copy_psnr"),
            ("decoded_copy_global_ssim", "decoded_copy_global_ssim"),
        )
    else:
        fields = (
            ("copy_latent_l1", "copy_latent_l1"),
            ("copy_latent_cosine_similarity", "copy_latent_cosine_similarity"),
            ("decoded_copy_psnr", "decoded_copy_psnr"),
            ("decoded_copy_global_ssim", "decoded_copy_global_ssim"),
        )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in report["rows"]:
        grouped[pair_key(row)].append(row)
    inconsistent: dict[str, list[str]] = {}
    for key, rows in grouped.items():
        failed = []
        for legacy, current in fields:
            field = current if current in rows[0] else legacy
            if field not in rows[0]:
                continue
            values = [float(row[field]) for row in rows]
            if max(values) - min(values) > 1e-7:
                failed.append(field)
        if failed:
            inconsistent[key] = failed
    return {
        "pair_count": len(grouped),
        "inconsistent_pair_count": len(inconsistent),
        "definition": "full-frame Copy metrics only; same-valid metrics are method-mask dependent",
        "details": inconsistent,
    }


def aggregate_legacy(report: dict, bootstrap_samples: int, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in report["rows"]:
        grouped[row["alignment"]].append(row)
    output = {}
    for offset, (method, rows) in enumerate(sorted(grouped.items())):
        psnr_delta = [
            float(row["decoded_warp_psnr"]) - float(row["decoded_copy_psnr"])
            for row in rows
        ]
        ssim_delta = [
            float(row["decoded_warp_global_ssim"])
            - float(row["decoded_copy_global_ssim"])
            for row in rows
        ]
        low, high = bootstrap_mean_interval(
            psnr_delta, bootstrap_samples, seed + offset
        )
        output[method] = {
            "pair_count": len(rows),
            "legacy_warp_latent_l1_mean": average(
                [float(row["warp_latent_l1"]) for row in rows]
            ),
            "legacy_copy_latent_l1_full_mean": average(
                [float(row["copy_latent_l1"]) for row in rows]
            ),
            "legacy_coverage_mass_mean": average(
                [float(row["latent_coverage"]) for row in rows]
            ),
            "decoded_psnr_delta_mean": average(psnr_delta),
            "decoded_psnr_delta_bootstrap_95ci": [low, high],
            "decoded_psnr_win_rate": sum(value > 0 for value in psnr_delta)
            / len(psnr_delta),
            "decoded_global_ssim_delta_mean": average(ssim_delta),
            "decoded_global_ssim_win_rate": sum(value > 0 for value in ssim_delta)
            / len(ssim_delta),
        }
    return output

def aggregate_current(report: dict, bootstrap_samples: int, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in report["rows"]:
        grouped[row["alignment"]].append(row)
    output = {}
    for offset, (method, rows) in enumerate(sorted(grouped.items())):
        valid_l1_delta = [
            float(row["warp_latent_l1_valid"])
            - float(row["copy_latent_l1_same_valid"])
            for row in rows
        ]
        valid_cosine_delta = [
            float(row["warp_latent_cosine_similarity_valid"])
            - float(row["copy_latent_cosine_similarity_same_valid"])
            for row in rows
        ]
        composite_l1_delta = [
            float(row["composite_latent_l1_full"])
            - float(row["copy_latent_l1_full"])
            for row in rows
        ]
        composite_psnr_delta = [
            float(row["decoded_composite_psnr"])
            - float(row["decoded_copy_psnr"])
            for row in rows
        ]
        zero_hole_psnr_delta = [
            float(row["decoded_warp_psnr"]) - float(row["decoded_copy_psnr"])
            for row in rows
        ]
        l1_low, l1_high = bootstrap_mean_interval(
            valid_l1_delta, bootstrap_samples, seed + 10 * offset
        )
        psnr_low, psnr_high = bootstrap_mean_interval(
            composite_psnr_delta, bootstrap_samples, seed + 10 * offset + 1
        )
        motion_bins = {}
        for label, lower, upper in (
            ("small", 0.0, 0.03),
            ("medium", 0.03, 0.06),
            ("large", 0.06, math.inf),
        ):
            selected = [
                row
                for row in rows
                if lower <= float(row["translation_norm_teacher_units"]) < upper
            ]
            if selected:
                motion_bins[label] = {
                    "pair_count": len(selected),
                    "translation_mean_teacher_units": average(
                        [float(row["translation_norm_teacher_units"]) for row in selected]
                    ),
                    "valid_l1_delta_mean": average(
                        [
                            float(row["warp_latent_l1_valid"])
                            - float(row["copy_latent_l1_same_valid"])
                            for row in selected
                        ]
                    ),
                    "decoded_composite_psnr_delta_mean": average(
                        [
                            float(row["decoded_composite_psnr"])
                            - float(row["decoded_copy_psnr"])
                            for row in selected
                        ]
                    ),
                    "decoded_composite_psnr_win_rate": sum(
                        float(row["decoded_composite_psnr"])
                        > float(row["decoded_copy_psnr"])
                        for row in selected
                    )
                    / len(selected),
                }
        output[method] = {
            "pair_count": len(rows),
            "binary_coverage_mean": average(
                [float(row["latent_coverage_binary"]) for row in rows]
            ),
            "valid_l1_delta_mean": average(valid_l1_delta),
            "valid_l1_delta_median": statistics.median(valid_l1_delta),
            "valid_l1_delta_bootstrap_95ci": [l1_low, l1_high],
            "valid_l1_win_rate": sum(value < 0 for value in valid_l1_delta)
            / len(valid_l1_delta),
            "valid_cosine_delta_mean": average(valid_cosine_delta),
            "valid_cosine_win_rate": sum(value > 0 for value in valid_cosine_delta)
            / len(valid_cosine_delta),
            "composite_full_l1_delta_mean": average(composite_l1_delta),
            "composite_full_l1_win_rate": sum(value < 0 for value in composite_l1_delta)
            / len(composite_l1_delta),
            "decoded_composite_psnr_delta_mean": average(composite_psnr_delta),
            "decoded_composite_psnr_delta_median": statistics.median(composite_psnr_delta),
            "decoded_composite_psnr_delta_bootstrap_95ci": [psnr_low, psnr_high],
            "decoded_composite_psnr_win_rate": sum(value > 0 for value in composite_psnr_delta)
            / len(composite_psnr_delta),
            "decoded_zero_hole_psnr_delta_mean": average(zero_hole_psnr_delta),
            "motion_bins": motion_bins,
        }
    return output


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text())
    schema_version = int(report["schema_version"])
    grain = validate_grain(report)
    consistency = copy_consistency(report)
    findings = []
    aggregates = {}
    evidence_status = "usable_with_declared_limitations"
    training_gate = None

    if schema_version == 1:
        aggregates = aggregate_legacy(
            report, args.bootstrap_samples, args.seed
        )
        findings.extend(
            [
                {
                    "severity": "critical",
                    "finding": "coverage is transformed splat mass, not binary occupancy",
                    "evidence": {
                        "identity_mass_reported_as_coverage": 1.0 - math.exp(-1.0),
                        "identity_false_hole_ratio": math.exp(-1.0),
                    },
                    "impact": "legacy coverage and hole-ratio claims are invalid",
                },
                {
                    "severity": "high",
                    "finding": "Warp and Copy latent metrics use different spatial support",
                    "evidence": "warp is coverage-masked while Copy is full-frame",
                    "impact": "legacy latent L1/cosine deltas are not comparable",
                },
                {
                    "severity": "high",
                    "finding": "pose-screened sample selection is not an unbiased benchmark",
                    "evidence": report["pose_status"],
                    "impact": "results cannot support GT feasibility or generalization",
                },
                {
                    "severity": "medium",
                    "finding": "decoded SSIM is global rather than standard windowed SSIM",
                    "evidence": "metric field is decoded_*_global_ssim",
                    "impact": "do not compare it with standard SSIM from prior work",
                },
            ]
        )
        evidence_status = "superseded_for_coverage_and_latent_copy_claims"
    elif schema_version == 2:
        aggregates = aggregate_current(
            report, args.bootstrap_samples, args.seed
        )
        best_method, best = max(
            aggregates.items(),
            key=lambda item: item[1]["decoded_composite_psnr_delta_mean"],
        )
        training_gate = {
            "passed": best["decoded_composite_psnr_delta_bootstrap_95ci"][0] > 0
            and best["decoded_composite_psnr_win_rate"] >= 0.7
            and best["valid_l1_win_rate"] >= 0.7,
            "criterion": "positive composite PSNR 95% CI, >=70% composite PSNR wins, and >=70% valid-L1 wins",
            "best_method_by_mean_composite_psnr": best_method,
            "observed": {
                "composite_psnr_delta_mean": best["decoded_composite_psnr_delta_mean"],
                "composite_psnr_delta_bootstrap_95ci": best[
                    "decoded_composite_psnr_delta_bootstrap_95ci"
                ],
                "composite_psnr_win_rate": best["decoded_composite_psnr_win_rate"],
                "valid_l1_win_rate": best["valid_l1_win_rate"],
            },
        }
        if not training_gate["passed"]:
            findings.append(
                {
                    "severity": "high",
                    "finding": "corrected estimated-pose screening does not pass the Student-training gate",
                    "evidence": training_gate,
                    "impact": "do not train the latent geometry Student from this evidence",
                }
            )
        findings.extend(
            [
                {
                    "severity": "high",
                    "finding": "poses remain MoGe-assisted estimates rather than ground truth",
                    "evidence": report["pose_status"],
                    "impact": "the negative result cannot distinguish VAE limits from pose error",
                },
                {
                    "severity": "medium",
                    "finding": "cosine gains do not transfer consistently to L1 or decoded quality",
                    "evidence": "see per-method valid cosine, valid L1, and composite PSNR win rates",
                    "impact": "cosine alone is not a sufficient feasibility criterion",
                },
                {
                    "severity": "medium",
                    "finding": "standard windowed SSIM and LPIPS are still unavailable",
                    "evidence": "decoded global SSIM is reported and decoded_lpips is null",
                    "impact": "decoded quality evidence is incomplete",
                },
            ]
        )
        evidence_status = (
            "corrected_screening_passed_with_estimated_pose_limitations"
            if training_gate["passed"]
            else "corrected_screening_negative_do_not_train_student"
        )
    else:
        raise ValueError(f"unsupported experiment schema version: {schema_version}")

    audit = {
        "schema_version": 1,
        "source": str(args.input),
        "source_experiment_schema_version": schema_version,
        "training_gate": training_gate,
        "evidence_status": evidence_status,
        "dataset_and_grain": grain,
        "copy_baseline_consistency": consistency,
        "aggregates": aggregates,
        "findings": findings,
        "safe_to_cite": {
            "decoded_full_frame_psnr_delta": True,
            "coverage_or_hole_ratio": schema_version == 2,
            "warp_vs_copy_same_mask_latent_delta": schema_version == 2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"evidence_status": evidence_status, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
