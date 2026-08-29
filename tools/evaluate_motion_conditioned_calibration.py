"""Evaluate motion-conditioned warp confidence for each target motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_reprojection_models import load_val_records, pick_records
from evaluate_warp_confidence_calibration import binary_auc, ece, threshold_curve
from train_motion_conditioned_reprojection_student import (
    MotionConditionedReprojectionStudent,
    encode_motion,
    project,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = MotionConditionedReprojectionStudent(width=int(config["width"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
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
    with torch.inference_mode():
        for record in records:
            sample_id = record["sample_id"]
            data = np.load(args.teacher / record["geometry"], allow_pickle=False)
            rgb = data["rgb"].astype(np.float32) / 255.0
            image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            points = torch.from_numpy(data["points"].astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
            teacher_mask = torch.from_numpy(data["mask"].astype(np.float32))[None, None].to(device)
            intrinsics = torch.from_numpy(data["intrinsics"].astype(np.float32)).unsqueeze(0).to(device)
            for motion in motions:
                motion_tensor = encode_motion(motion["yaw"], motion["forward"], device)
                pred = model(image, motion_tensor)
                confidence = torch.sigmoid(pred["warp_logits"]).squeeze().detach().cpu().numpy().astype(np.float32)
                _, valid = project(points, intrinsics, yaw=motion["yaw"], forward=motion["forward"])
                label = (valid * teacher_mask).squeeze().detach().cpu().numpy().astype(bool)
                valid_pixels = teacher_mask.squeeze().detach().cpu().numpy().astype(bool) & np.isfinite(confidence)
                scores = np.nan_to_num(confidence[valid_pixels], nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
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
    by_motion = {}
    for motion in sorted({r["motion"] for r in rows}):
        items = [r for r in rows if r["motion"] == motion]
        by_motion[motion] = {
            "num_cases": len(items),
            "auc_mean": float(np.nanmean([r["auc"] for r in items])),
            "positive_rate_mean": float(np.mean([r["positive_rate"] for r in items])),
            "confidence_mean": float(np.mean([r["confidence_mean"] for r in items])),
        }
    summary = {
        "teacher": args.teacher.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_val_loss": state.get("val_loss"),
        "num_samples": len(records),
        "num_cases": len(rows),
        "num_pixels": int(scores.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "confidence_mean": float(scores.mean()) if scores.size else 0.0,
        "auc_mean": float(np.nanmean([r["auc"] for r in rows])) if rows else float("nan"),
        "auc_global": binary_auc(scores, labels) if scores.size else float("nan"),
        "ece": ece_value,
        "by_motion": by_motion,
        "bins": bins,
        "threshold_curve": threshold_curve(scores, labels) if scores.size else [],
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
