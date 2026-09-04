#!/usr/bin/env python3
"""Benchmark the read-only M1 to existing M2 bridge without changing M2."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_geometry_head import LatentGeometryHead
from latent_m2_bridge import export_latent_geometry_to_m2
from latent_motion_confidence import LatentMotionConfidence


def timed(fn, warmup: int, repeat: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-repo", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")

    sys.path.insert(0, str(args.m2_repo.resolve()))
    from geosparse.geometry import RelativePose, geometry_pose_candidate

    device = torch.device("cuda:0")
    geometry = LatentGeometryHead().to(device, dtype=torch.bfloat16).eval()
    geometry.load_state_dict(torch.load(args.geometry_checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    confidence = LatentMotionConfidence().to(device, dtype=torch.bfloat16).eval()
    confidence.load_state_dict(torch.load(args.confidence_checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    batch_rows = []
    for batch in (1, 4):
        latent = torch.randn(batch, 16, 44, 80, device=device, dtype=torch.bfloat16)
        intrinsics = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], device=device, dtype=torch.bfloat16).unsqueeze(0).expand(batch, -1, -1)
        motion = torch.zeros(batch, 6, device=device, dtype=torch.bfloat16)
        out = geometry(latent, intrinsics)
        logits = confidence(latent, out.latent_depth.to(torch.bfloat16), out.latent_valid_logits.to(torch.bfloat16), motion)
        out = out.with_confidence(logits)
        bridge_input = out if batch == 1 else geometry(latent[:1], intrinsics[:1])
        payload = export_latent_geometry_to_m2(bridge_input)
        source_history = latent.float().cpu().numpy()[:, :, None]
        pose = RelativePose(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, {"source": "benchmark"})

        def m1_step():
            return geometry(latent, intrinsics)

        def confidence_step():
            current = geometry(latent, intrinsics)
            return confidence(latent, current.latent_depth.to(torch.bfloat16), current.latent_valid_logits.to(torch.bfloat16), motion)

        def complete_m1_step():
            current = geometry(latent, intrinsics)
            return current.with_confidence(confidence(latent, current.latent_depth.to(torch.bfloat16), current.latent_valid_logits.to(torch.bfloat16), motion))

        def bridge_step():
            return [export_latent_geometry_to_m2(bridge_input) for _ in range(batch)]

        def m2_step():
            return [geometry_pose_candidate(source_history[i:i + 1], payload.geometry, pose, horizons=(4, 8, 12), reference_horizon_rgb_frames=12.0, name="latency_benchmark") for i in range(batch)]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        row = {"batch": batch, "m1_geometry": timed(m1_step, args.warmup, args.repeat, device), "m1_confidence": timed(confidence_step, args.warmup, args.repeat, device), "m1_complete": timed(complete_m1_step, args.warmup, args.repeat, device), "bridge_export": timed(bridge_step, args.warmup, args.repeat, device), "m2_candidate_renderer": timed(m2_step, args.warmup, args.repeat, device)}
        row["peak_allocated_mib"] = float(torch.cuda.max_memory_allocated() / 2**20)
        row["peak_reserved_mib"] = float(torch.cuda.max_memory_reserved() / 2**20)
        batch_rows.append(row)

    chunk_rows = []
    for chunk in (1, 3, 5):
        latent = torch.randn(1, 16, chunk, 44, 80, device=device, dtype=torch.bfloat16)
        intrinsics = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]], device=device, dtype=torch.bfloat16)

        def chunk_m1():
            return [geometry(latent[:, :, index], intrinsics) for index in range(chunk)]

        def chunk_m2():
            outputs = [geometry(latent[:, :, index], intrinsics) for index in range(chunk)]
            return [export_latent_geometry_to_m2(item) for item in outputs]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        chunk_rows.append({"chunk_frames": chunk, "m1_per_chunk": timed(chunk_m1, args.warmup, args.repeat, device), "m1_m2_bridge_per_chunk": timed(chunk_m2, args.warmup, args.repeat, device), "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20), "peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20)})

    m2_status = subprocess.run(["git", "status", "--short"], cwd=args.m2_repo, check=True, capture_output=True, text=True).stdout.strip()
    if m2_status:
        raise RuntimeError("M2 became dirty during read-only benchmark")
    report = {"schema_version": 1, "stage": "M1 to M2 latency, memory, batch and chunk benchmark", "m2_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.m2_repo, text=True).strip(), "m2_dirty_after": False, "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "latent_shape": "[B,16,T,44,80]", "geometry_checkpoint": str(args.geometry_checkpoint), "confidence_checkpoint": str(args.confidence_checkpoint), "batch_rows": batch_rows, "chunk_rows": chunk_rows, "scope": "M1 and existing M2 candidate/renderer interface; excludes Frozen VAE, DiT and causal video rollout"}
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("实验已完成", flush=True)


if __name__ == "__main__":
    main()
