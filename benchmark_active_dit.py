"""Microbenchmark packed active-query execution for a DiT-like stack.

This is not Matrix-Game latency.  It measures whether reducing the number of
query tokens produces real wall-clock savings on the local GPU when full-scene
K/V is already resident, matching the proposed Reuse/Repair/Regenerate runtime.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ActiveQueryBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.norm_mlp = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * mlp_ratio, bias=False)
        self.fc2 = nn.Linear(width * mlp_ratio, width, bias=False)

    @torch.inference_mode()
    def make_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = context.shape
        value = self.norm_kv(context)
        k = self.k_proj(value).reshape(batch, tokens, self.heads, self.head_dim)
        v = self.v_proj(value).reshape(batch, tokens, self.heads, self.head_dim)
        return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    @torch.inference_mode()
    def forward(
        self, query: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch, tokens, width = query.shape
        residual = query
        q = self.q_proj(self.norm_q(query)).reshape(
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        k, v = kv
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        query = residual + self.out_proj(attended)
        query = query + self.fc2(F.gelu(self.fc1(self.norm_mlp(query))))
        return query


@torch.inference_mode()
def run_stack(
    query: torch.Tensor,
    blocks: nn.ModuleList,
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    for block, kv in zip(blocks, kv_cache):
        query = block(query, kv)
    return query


def benchmark(call, warmup: int, repeat: int, inner: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(inner):
            call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0 / inner)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--inner", type=int, default=10,
                        help="Model evaluations per synchronization/timed sample")
    parser.add_argument("--preheat", type=int, default=50,
                        help="Full-token calls before any timed ratio to stabilize GPU clocks")
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]
    )
    args = parser.parse_args()

    if min(args.tokens, args.width, args.heads, args.layers, args.repeat, args.inner) <= 0:
        raise ValueError("positive dimensions, repeat and inner required")
    if not all(0 < ratio <= 1 for ratio in args.ratios) or 1.0 not in args.ratios:
        raise ValueError("ratios must be in (0, 1] and include 1.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.width % args.heads:
        raise ValueError("width must be divisible by heads")
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    blocks = nn.ModuleList(
        [ActiveQueryBlock(args.width, args.heads, args.mlp_ratio) for _ in range(args.layers)]
    ).to(device=device, dtype=dtype).eval()
    context = torch.randn(1, args.tokens, args.width, device=device, dtype=dtype)
    kv_cache = [block.make_kv(context) for block in blocks]

    # Laptop GPUs can change clocks substantially during the first seconds.
    # Stabilize before comparing token ratios so the full-token row is not
    # unfairly penalized by a cold/low-clock measurement.
    for _ in range(max(0, args.preheat)):
        run_stack(context, blocks, kv_cache)
    torch.cuda.synchronize()

    active_counts = [max(1, round(args.tokens * ratio)) for ratio in args.ratios]
    queries = [context[:, :count].clone() for count in active_counts]
    times_by_ratio: list[list[float]] = [[] for _ in queries]
    rng = random.Random(7)

    # Interleave ratios in a deterministic shuffled order.  This prevents GPU
    # boost/thermal changes over time from systematically favoring later rows.
    for round_index in range(args.warmup + args.repeat):
        order = list(range(len(queries)))
        rng.shuffle(order)
        for index in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.inner):
                run_stack(queries[index], blocks, kv_cache)
            torch.cuda.synchronize()
            if round_index >= args.warmup:
                times_by_ratio[index].append(
                    (time.perf_counter() - start) * 1000.0 / args.inner
                )

    rows = []
    for ratio, active_tokens, times in zip(args.ratios, active_counts, times_by_ratio):
        rows.append(
            {
                "active_ratio": active_tokens / args.tokens,
                "active_tokens": active_tokens,
                "times_ms": [round(value, 3) for value in times],
                "mean_ms": round(float(np.mean(times)), 3),
                "median_ms": round(float(np.median(times)), 3),
                "p95_ms": round(float(np.percentile(times, 95)), 3),
            }
        )

    full_median = next(row["median_ms"] for row in rows if row["active_ratio"] == 1.0)
    for row in rows:
        row["relative_latency"] = round(row["median_ms"] / full_median, 4)
        row["speedup_vs_full"] = round(full_median / row["median_ms"], 3)

    report = {
        "scope": "DiT-like active-query microbenchmark; not Matrix-Game end-to-end latency",
        "device": torch.cuda.get_device_name(),
        "dtype": str(dtype),
        "tokens": args.tokens,
        "width": args.width,
        "heads": args.heads,
        "layers": args.layers,
        "mlp_ratio": args.mlp_ratio,
        "inner_evaluations_per_sample": args.inner,
        "preheat": args.preheat,
        "warmup_rounds": args.warmup,
        "repeat_rounds": args.repeat,
        "seed": 7,
        "kv_policy": "full-scene K/V precomputed and resident; active Q packed",
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "active_dit_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    x = [100 * row["active_ratio"] for row in rows]
    y = [row["median_ms"] for row in rows]
    ideal = [full_median * r["active_ratio"] for r in rows]
    fig, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    axis.plot(x, y, marker="o", linewidth=2, label="Measured packed active-Q")
    axis.plot(x, ideal, linestyle="--", color="gray", label="Ideal linear scaling")
    axis.set_xlabel("Active query tokens (%)")
    axis.set_ylabel(f"One {args.layers}-block DiT-like evaluation (ms)")
    axis.set_title("Packed active-Q proxy latency\nRandom weights; full-scene K/V already resident")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    axis.legend()
    axis.invert_xaxis()
    fig.savefig(args.output / "active_dit_scaling.png", dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
