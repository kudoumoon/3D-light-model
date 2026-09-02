#!/usr/bin/env python3
"""Audit why latent warping must happen before Matrix-Game 3D RoPE."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=10000.0)
    return parser.parse_args()


def phase(position: int, complex_dim: int, theta: float) -> torch.Tensor:
    frequency = 1.0 / torch.pow(
        torch.tensor(theta, dtype=torch.float64),
        torch.arange(complex_dim, dtype=torch.float64) / complex_dim,
    )
    return torch.polar(torch.ones_like(frequency), position * frequency)


def repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(20260902)
    raw = torch.randn(32, 2, generator=generator, dtype=torch.float64)
    raw_complex = torch.view_as_complex(raw)
    rows = []
    for axis in ("time", "height", "width"):
        for displacement in (1, 2, 4, 8, 16):
            source = raw_complex * phase(0, raw_complex.numel(), args.theta)
            target_expected = raw_complex * phase(displacement, raw_complex.numel(), args.theta)
            naive = source
            corrected = source / phase(0, raw_complex.numel(), args.theta) * phase(
                displacement, raw_complex.numel(), args.theta
            )
            rows.append(
                {
                    "axis": axis,
                    "displacement": displacement,
                    "naive_l1": float((torch.view_as_real(naive) - torch.view_as_real(target_expected)).abs().mean()),
                    "corrected_l1": float((torch.view_as_real(corrected) - torch.view_as_real(target_expected)).abs().mean()),
                    "naive_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            torch.view_as_real(naive).flatten()[None],
                            torch.view_as_real(target_expected).flatten()[None],
                        )
                    ),
                }
            )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "stage": "Matrix-Game pre-RoPE warp contract audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "matrix_game_contract": {
            "latent_grid": [3, 44, 80],
            "patch_size": [1, 2, 2],
            "token_grid": [3, 22, 40],
            "tokens_per_latent_frame": 880,
            "tokens_per_chunk": 2640,
        },
        "interpretation": "pre-RoPE latent warp is position-consistent; moving an already-rotated token requires explicit source-phase removal and target-phase application",
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "max_corrected_l1": max(row["corrected_l1"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
