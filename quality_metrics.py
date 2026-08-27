"""Offline RGB-threshold diagnostics; these are not perceptual safety guarantees."""

from __future__ import annotations

import numpy as np


def tile_pass_fraction(error, mask, threshold: float, tile: int = 16) -> float:
    if tile <= 0 or error.shape != mask.shape:
        raise ValueError("positive tile and matching 2D arrays required")
    passed = total = 0
    h, w = mask.shape
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            valid = mask[y:y + tile, x:x + tile]
            values = error[y:y + tile, x:x + tile]
            total += 1
            passed += bool(valid.mean() >= 0.95 and values[valid].mean() <= threshold)
    return passed / max(1, total)


def tile_candidate_oracle_fraction(copy_error, warp_error, warp_mask,
                                   threshold: float, tile: int = 16) -> float:
    """Choose one candidate per tile, NOT a per-pixel mosaic of both candidates.

    A warp tile must have >=95% support and pass covered-pixel mean RGB MAE;
    copy has full support. The target is needed to compute both errors.
    Partial bottom/right tiles each count once, matching the recorded metric.
    """
    if tile <= 0 or copy_error.shape != warp_error.shape or copy_error.shape != warp_mask.shape:
        raise ValueError("positive tile and matching 2D arrays required")
    passed = total = 0
    h, w = warp_mask.shape
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            c = copy_error[y:y + tile, x:x + tile]
            e = warp_error[y:y + tile, x:x + tile]
            m = warp_mask[y:y + tile, x:x + tile]
            total += 1
            copy_ok = c.mean() <= threshold
            warp_ok = m.mean() >= 0.95 and e[m].mean() <= threshold
            passed += bool(copy_ok or warp_ok)
    return passed / max(1, total)
