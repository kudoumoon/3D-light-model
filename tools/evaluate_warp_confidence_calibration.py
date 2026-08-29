"""Evaluate warp-confidence calibration against teacher projected-valid labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_reprojection_student_warp import project
from compare_reprojection_models import load_val_records, pick_records


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.reshape(-1)
    labels = labels.reshape(-1).astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def ece(scores: np.ndarray, labels: np.ndarray, bins: int) -> tuple[float, list[dict]]:
    scores = scores.reshape(-1)
    labels = labels.reshape(-1).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(scores)
    rows = []
    value = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": 0})
            continue
        conf = float(scores[mask].mean())
        acc = float(labels[mask].mean())
        gap = abs(conf - acc)
        value += (count / total) * gap
        rows.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": count, "confidence": conf, "accuracy": acc, "gap": gap})
    return float(value), rows


def threshold_curve(scores: np.ndarray, labels: np.ndarray) -> list[dict]:
    scores = scores.reshape(-1)
    labels = labels.reshape(-1).astype(np.float32)
    rows = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = scores >= threshold
        count = int(keep.sum())
        rows.append({
            "threshold": threshold,
            "kept_ratio": float(keep.mean()),
            "projected_valid_rate": float(labels[keep].mean()) if count else 0.0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    motions = [
        {"name": "yaw_p5_fwd10", "yaw": 5.0, "forward": 0.10},
        {"name": "yaw_m5_fwd10", "yaw": -5.0, "forward": 0.10},
        {"name": "yaw_p10_fwd10", "yaw": 10.0, "forward": 0.10},
        {"name": "yaw_m10_fwd10", "yaw": -10.0, "forward": 0.10},
    ]
    records = pick_records(load_val_records(args.teacher), args.per_scene)
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    all_scores = []
    all_labels = []
    for record in records:
        sample_id = record["sample_id"]
        teacher_data = np.load(args.teacher / record["geometry"], allow_pickle=False)
        student_data = np.load(args.student_eval / "predictions" / sample_id / "geometry.npz", allow_pickle=False)
        points = torch.from_numpy(teacher_data["points"].astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        teacher_mask = torch.from_numpy(teacher_data["mask"].astype(np.float32))[None, None]
        intrinsics = torch.from_numpy(teacher_data["intrinsics"].astype(np.float32)).unsqueeze(0)
        confidence = student_data["warp_confidence"].astype(np.float32)
        finite = np.isfinite(confidence)
        confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        for motion in motions:
            _, valid = project(points, intrinsics, yaw=motion["yaw"], forward=motion["forward"])
            label = (valid * teacher_mask).squeeze().numpy().astype(bool)
            valid_pixels = teacher_mask.squeeze().numpy().astype(bool) & finite
            scores = confidence[valid_pixels]
            labels = label[valid_pixels]
            if scores.size == 0:
                continue
            row = {
                "sample_id": sample_id,
                "scene": record["scene"],
                "motion": motion["name"],
                "num_pixels": int(scores.size),
                "positive_rate": float(labels.mean()),
                "confidence_mean": float(scores.mean()),
                "auc": binary_auc(scores, labels),
            }
            rows.append(row)
            all_scores.append(scores)
            all_labels.append(labels)

    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=bool)
    ece_value, bins = ece(scores, labels, args.bins) if scores.size else (float("nan"), [])
    summary = {
        "teacher": args.teacher.as_posix(),
        "student_eval": args.student_eval.as_posix(),
        "num_samples": len(records),
        "num_cases": len(rows),
        "num_pixels": int(scores.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "confidence_mean": float(scores.mean()) if scores.size else 0.0,
        "auc_mean": float(np.nanmean([r["auc"] for r in rows])) if rows else float("nan"),
        "auc_global": binary_auc(scores, labels) if scores.size else float("nan"),
        "ece": ece_value,
        "bins": bins,
        "threshold_curve": threshold_curve(scores, labels) if scores.size else [],
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
