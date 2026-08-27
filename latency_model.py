"""Accounting assumptions for a structural proxy, never an end-to-end timer."""

from __future__ import annotations

from bisect import bisect_left
import math


def lookup_latency(rows: list[dict], ratio: float, policy: str = "floor") -> float:
    """Interpolate measured medians, with an explicit unmeasured low-ratio policy.

    floor: hold the smallest measured positive-ratio median below that ratio.
    legacy_linear: connect the smallest point to (0, 0); optimistic and untested.
    Exactly zero assumes the entire operation is bypassed. Neither policy is a
    measured latency or a guaranteed bound outside the sampled ratios.
    """
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be finite and in [0, 1]")
    if policy not in {"floor", "legacy_linear"}:
        raise ValueError("unknown low-ratio policy")
    samples = sorted((float(r["active_ratio"]), float(r["median_ms"])) for r in rows)
    if not samples or samples[0][0] <= 0 or samples[-1][0] != 1:
        raise ValueError("positive samples including the full-ratio baseline required")
    if any(not math.isfinite(x) or not math.isfinite(t) or t < 0 for x, t in samples):
        raise ValueError("samples must be finite and latency nonnegative")
    if len({x for x, _ in samples}) != len(samples):
        raise ValueError("sample ratios must be unique")
    if ratio == 0:
        return 0.0
    if ratio <= samples[0][0]:
        return samples[0][1] if policy == "floor" else samples[0][1] * ratio / samples[0][0]
    pos = bisect_left([x for x, _ in samples], ratio)
    x0, y0 = samples[pos - 1]
    x1, y1 = samples[pos]
    return y0 + (y1 - y0) * (ratio - x0) / (x1 - x0)


def interval_cost(rows, f0, geometry_ms=0.0, warp_ms=0.0, interval=8, policy="floor"):
    if interval < 1 or not 0 <= f0 <= 1 or min(geometry_ms, warp_ms) < 0:
        raise ValueError("invalid accounting parameters")
    baseline = 4 * lookup_latency(rows, 1.0, policy)
    reuse_frame = warp_ms + 4 * lookup_latency(rows, 1 - f0, policy)
    average = (baseline + geometry_ms + (interval - 1) * reuse_frame) / interval
    return {"baseline_ms_proxy": baseline, "average_ms_proxy": average,
            "speedup_proxy": baseline / average}
