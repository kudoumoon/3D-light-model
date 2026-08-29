"""Evaluate and export a motion-conditioned reprojection student checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from train_motion_conditioned_reprojection_student import (
    MotionConditionedReprojectionStudent,
    TeacherDataset,
    average,
    compute_loss,
    encode_motion,
    export_prediction,
)


@torch.inference_mode()
def benchmark_ms(
    model: MotionConditionedReprojectionStudent,
    sample: dict,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    image = sample["image"].unsqueeze(0).to(device)
    motion = encode_motion(5.0, 0.10, device)
    if device.type == "cuda":
        for _ in range(warmup):
            _ = model(image, motion)
        torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(image, motion)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))
    else:
        for _ in range(warmup):
            _ = model(image, motion)
        timings = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(image, motion)
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(statistics.median(timings)),
        "mean_ms": float(statistics.fmean(timings)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "repeat": repeat,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = MotionConditionedReprojectionStudent(width=int(config["width"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    val_set = TeacherDataset(args.teacher, "val")
    weights = config["loss_weights"]
    rows = []
    for index in range(len(val_set)):
        sample = val_set[index]
        _, metrics = compute_loss(model, sample, device, weights)
        rows.append(metrics)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(parents=True, exist_ok=True)
    exports = []
    for index in range(len(val_set)):
        sample = val_set[index]
        pred_dir = args.output / "predictions" / str(sample["id"])
        export_prediction(model, sample, device, pred_dir)
        exports.append(
            {
                "sample_id": sample["id"],
                "scene": sample["scene"],
                "geometry": pred_dir.relative_to(args.output).as_posix() + "/geometry.npz",
            }
        )

    latency = benchmark_ms(model, val_set[0], device, args.warmup, args.repeat)
    summary = {
        "teacher": args.teacher.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_val_loss": state.get("val_loss"),
        "num_samples": len(val_set),
        "metrics": average(rows),
        "inference_latency": latency,
        "predictions": exports,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
