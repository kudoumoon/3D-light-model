"""Explicit RGB-geometry to latent-grid alignment operators.

These operators are experimental conditions, not interchangeable preprocessing.
Every operator returns a depth/valid pair on the exact spatial VAE grid so the
selection criterion can be target-latent warp quality rather than depth error.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


AlignmentMethod = Literal["center", "average", "median", "minimum", "confidence_weighted"]


def _validate(depth: torch.Tensor, valid: torch.Tensor, output_size: tuple[int, int]) -> None:
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must be [B, 1, H, W]")
    if valid.shape != depth.shape:
        raise ValueError("valid must have the same shape as depth")
    if min(output_size) <= 0:
        raise ValueError("output_size must be positive")
    in_height, in_width = depth.shape[-2:]
    out_height, out_width = output_size
    if in_height % out_height or in_width % out_width:
        raise ValueError("RGB geometry size must be divisible by latent grid size")


def align_depth_to_latent(
    depth: torch.Tensor,
    valid: torch.Tensor,
    output_size: tuple[int, int],
    method: AlignmentMethod,
    confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool RGB depth on non-overlapping VAE receptive-field proxy cells."""

    _validate(depth, valid, output_size)
    if method == "confidence_weighted" and (confidence is None or confidence.shape != depth.shape):
        raise ValueError("confidence_weighted alignment requires confidence matching depth")
    batch, _, in_height, in_width = depth.shape
    out_height, out_width = output_size
    scale_h, scale_w = in_height // out_height, in_width // out_width
    patches = depth.view(batch, 1, out_height, scale_h, out_width, scale_w).permute(0, 1, 2, 4, 3, 5).reshape(batch, 1, out_height, out_width, -1)
    support = valid.view(batch, 1, out_height, scale_h, out_width, scale_w).permute(0, 1, 2, 4, 3, 5).reshape_as(patches).bool()
    # Invalid teacher pixels must not leak NaNs through masked reductions.
    patches = torch.where(support, torch.nan_to_num(patches), torch.zeros_like(patches))
    output_valid = support.any(dim=-1).to(depth.dtype)
    if method == "center":
        centre = patches.shape[-1] // 2
        value = patches[..., centre]
        centre_valid = support[..., centre]
        # A missing centre is a failed cell, not an invitation to silently use a
        # different rule.  This keeps the condition experimentally identifiable.
        return torch.where(centre_valid, value, torch.zeros_like(value)), centre_valid.to(depth.dtype)
    if method == "average":
        denom = support.sum(dim=-1).clamp_min(1)
        value = (patches * support).sum(dim=-1) / denom
    elif method == "median":
        value = patches.masked_fill(~support, torch.inf).sort(dim=-1).values
        rank = ((support.sum(dim=-1).clamp_min(1) - 1) // 2).long().unsqueeze(-1)
        value = value.gather(-1, rank).squeeze(-1)
    elif method == "minimum":
        value = patches.masked_fill(~support, torch.inf).amin(dim=-1)
    elif method == "confidence_weighted":
        weights = confidence.view(batch, 1, out_height, scale_h, out_width, scale_w).permute(0, 1, 2, 4, 3, 5).reshape_as(patches)
        weights = weights.clamp_min(0) * support
        value = (patches * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)
    else:
        raise ValueError(f"unsupported alignment method: {method}")
    return torch.where(output_valid.bool(), value, torch.zeros_like(value)), output_valid


class LearnedGeometryPooling(nn.Module):
    """Learn depth-aware weights inside each RGB-to-latent cell.

    This is a single controlled P3 condition.  It must be trained only against
    the same target-latent warp objective as the fixed pooling baselines.
    """

    def __init__(self, cell_height: int, cell_width: int, hidden: int = 32) -> None:
        super().__init__()
        if min(cell_height, cell_width, hidden) <= 0:
            raise ValueError("pool dimensions must be positive")
        self.cell_height = cell_height
        self.cell_width = cell_width
        self.score = nn.Sequential(
            nn.Conv2d(3, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, depth: torch.Tensor, valid: torch.Tensor, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if depth.shape != valid.shape or edge.shape != depth.shape:
            raise ValueError("depth, valid and edge must share shape [B,1,H,W]")
        height, width = depth.shape[-2:]
        if height % self.cell_height or width % self.cell_width:
            raise ValueError("input must divide exactly into configured pooling cells")
        logits = self.score(torch.cat((depth, valid, edge), dim=1))
        out_height, out_width = height // self.cell_height, width // self.cell_width
        def cellify(value: torch.Tensor) -> torch.Tensor:
            return value.view(value.shape[0], value.shape[1], out_height, self.cell_height, out_width, self.cell_width).permute(0, 1, 2, 4, 3, 5).reshape(value.shape[0], value.shape[1], out_height, out_width, -1)
        values, support, logits = cellify(depth), cellify(valid).bool(), cellify(logits)
        weights = torch.softmax(logits.masked_fill(~support, -torch.inf), dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        output_valid = support.any(dim=-1).to(depth.dtype)
        return (values * weights).sum(dim=-1), output_valid


def depth_edge_map(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """A normalized RGB-space depth boundary cue for learned pooling."""

    dx = F.pad((depth[..., :, 1:] - depth[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((depth[..., 1:, :] - depth[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = (dx + dy) * valid
    denom = edge.flatten(1).quantile(0.95, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (edge / denom).clamp(0, 1)
