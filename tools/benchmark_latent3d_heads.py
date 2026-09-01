#!/usr/bin/env python3
"""Benchmark the incremental M1 heads on an existing world-model latent."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_geometry_head import LatentGeometryHead
from latent_motion_confidence import LatentMotionConfidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=44)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def repository_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    function: Callable[[], object],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            function()
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.height, args.width, args.warmup, args.repeats) <= 0:
        raise ValueError("batch/grid/warmup/repeats values must be positive")
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("CPU benchmark is fixed to float32")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    geometry = LatentGeometryHead().to(device=device, dtype=dtype).eval()
    confidence = LatentMotionConfidence().to(device=device, dtype=dtype).eval()
    latent = torch.randn(
        args.batch_size, 16, args.height, args.width, device=device, dtype=dtype
    )
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).repeat(args.batch_size, 1, 1)
    motion = torch.zeros(args.batch_size, 6, device=device, dtype=dtype)
    with torch.inference_mode():
        geometry_output = geometry(latent, intrinsics)

    def geometry_step() -> object:
        return geometry(latent, intrinsics)

    def confidence_step() -> object:
        return confidence(
            latent,
            geometry_output.latent_depth,
            geometry_output.latent_valid_logits,
            motion,
        )

    def complete_step() -> object:
        output = geometry(latent, intrinsics)
        logits = confidence(
            latent, output.latent_depth, output.latent_valid_logits, motion
        )
        return output.with_confidence(logits)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latency = {
        "geometry_head": benchmark(geometry_step, device, args.warmup, args.repeats),
        "motion_confidence_head": benchmark(
            confidence_step, device, args.warmup, args.repeats
        ),
        "complete_m1_incremental": benchmark(
            complete_step, device, args.warmup, args.repeats
        ),
    }
    runtime = {
        "device": str(device),
        "dtype": str(dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(properties.major_minor)
            if hasattr(properties, "major_minor")
            else [properties.major, properties.minor],
        }
    report = {
        "schema_version": 1,
        "stage": "M1 incremental head benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "config": vars(args) | {"output": str(args.output)},
        "parameters": {
            "geometry_head": sum(parameter.numel() for parameter in geometry.parameters()),
            "motion_confidence_head": sum(
                parameter.numel() for parameter in confidence.parameters()
            ),
            "complete_m1_incremental": sum(
                parameter.numel()
                for model in (geometry, confidence)
                for parameter in model.parameters()
            ),
        },
        "runtime": runtime,
        "latency": latency,
        "interpretation": "CPU numbers are diagnostic only; paper latency requires an idle H100 and the same protocol.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "latency": latency}, indent=2))


if __name__ == "__main__":
    main()
