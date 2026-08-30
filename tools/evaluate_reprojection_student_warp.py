"""Evaluate and export a trained v3 reprojection student checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_reprojection_student_warp import (
    ReprojectionStudentWarp,
    TeacherDataset,
    compute_loss,
    export_prediction,
)


ROOT = Path(__file__).resolve().parents[1]


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=ROOT / "runs/teacher_moge3_video_384")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = ReprojectionStudentWarp(width=int(config["width"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = TeacherDataset(args.teacher, args.split)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher.as_posix(),
                "checkpoint": args.checkpoint.as_posix(),
                "split": args.split,
                "checkpoint_epoch": state["epoch"],
                "checkpoint_val_loss": state["val_loss"],
                "warmup": args.warmup,
                "repeat": args.repeat,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = []
    prediction_rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            _, metrics = compute_loss(model, sample, device, config["loss_weights"])
            image = sample["image"].unsqueeze(0).to(device)
            for _ in range(args.warmup):
                _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                _ = model(image)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
            row = {
                "sample_id": sample["id"],
                "scene": sample["scene"],
                **metrics,
                "inference_ms_median": float(np.median(times)),
                "inference_ms_p95": float(np.percentile(times, 95)),
            }
            rows.append(row)
            pred_dir = args.output / "predictions" / str(sample["id"])
            export_prediction(model, sample, device, pred_dir)
            prediction_rows.append(
                {
                    "sample_id": sample["id"],
                    "scene": sample["scene"],
                    "geometry": pred_dir.relative_to(args.output).as_posix() + "/geometry.npz",
                }
            )
            with (args.output / "per_sample.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(rows),
        "loss": mean(rows, "loss"),
        "point": mean(rows, "point"),
        "mask": mean(rows, "mask"),
        "normal": mean(rows, "normal"),
        "edge": mean(rows, "edge"),
        "projection": mean(rows, "projection"),
        "warp": mean(rows, "warp"),
        "occupancy": mean(rows, "occupancy"),
        "warp_target_mean": mean(rows, "warp_target_mean"),
        "inference_ms_median_mean": mean(rows, "inference_ms_median"),
        "inference_ms_p95_mean": mean(rows, "inference_ms_p95"),
        "predictions": prediction_rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
