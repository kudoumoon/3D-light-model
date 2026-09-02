"""Reprojection-oriented surface selection on an RGB-to-latent cell grid.

Unlike mean depth pooling, the selector learns a categorical distribution over
the valid RGB geometry samples inside each latent cell.  It also exposes the
distribution entropy and depth dispersion, so mixed foreground/background cells
can be rejected by a confidence head instead of being assigned false certainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from latent_geometry_alignment import depth_edge_map


@dataclass(frozen=True)
class SurfaceAlignmentOutput:
    depth: torch.Tensor
    valid: torch.Tensor
    ambiguity: torch.Tensor
    relative_dispersion: torch.Tensor
    support_fraction: torch.Tensor
    weights: torch.Tensor


def _cellify(value: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    batch, channels, height, width = value.shape
    out_height, out_width = output_size
    if height % out_height or width % out_width:
        raise ValueError("RGB geometry must divide exactly into the latent grid")
    cell_height, cell_width = height // out_height, width // out_width
    return (
        value.view(batch, channels, out_height, cell_height, out_width, cell_width)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch, channels, out_height, out_width, cell_height * cell_width)
    )


class ReprojectionOptimalSurfaceSelector(nn.Module):
    """Select one dominant projective surface per latent cell.

    ``latent`` supplies the cell query and RGB depth/edge/confidence samples
    supply candidate keys.  The output remains differentiable and is intended
    to be trained using target-latent reprojection quality, not depth error alone.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden: int = 32,
        temperature: float = 0.15,
    ) -> None:
        super().__init__()
        if min(latent_channels, hidden) <= 0 or temperature <= 0:
            raise ValueError("channels, hidden and temperature must be positive")
        self.temperature = temperature
        self.pixel_key = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, hidden)
        )
        self.latent_query = nn.Conv2d(latent_channels, hidden, 1)
        self.score_bias = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        depth: torch.Tensor,
        valid: torch.Tensor,
        latent: torch.Tensor,
        confidence: torch.Tensor,
    ) -> SurfaceAlignmentOutput:
        if depth.ndim != 4 or depth.shape[1] != 1 or valid.shape != depth.shape:
            raise ValueError("depth and valid must share shape [B,1,H,W]")
        if confidence.shape != depth.shape:
            raise ValueError("confidence must match depth")
        if latent.ndim != 4 or latent.shape[0] != depth.shape[0]:
            raise ValueError("latent must have shape [B,C,H_latent,W_latent]")

        output_size = latent.shape[-2:]
        depth_cells = _cellify(depth, output_size)
        valid_cells = _cellify(valid, output_size).bool()
        confidence_cells = _cellify(confidence.clamp(0, 1), output_size)
        edge_cells = _cellify(depth_edge_map(depth, valid), output_size)

        safe_depth = torch.where(valid_cells, depth_cells.clamp_min(1e-6), torch.ones_like(depth_cells))
        log_depth = safe_depth.log()
        denom = valid_cells.sum(dim=-1, keepdim=True).clamp_min(1)
        local_mean = (log_depth * valid_cells).sum(dim=-1, keepdim=True) / denom
        relative_log_depth = log_depth - local_mean
        pixel_features = torch.stack(
            (
                relative_log_depth[:, 0],
                edge_cells[:, 0],
                confidence_cells[:, 0],
                valid_cells[:, 0].to(depth.dtype),
            ),
            dim=-1,
        )

        keys = self.pixel_key(pixel_features)
        query = self.latent_query(latent).permute(0, 2, 3, 1).unsqueeze(-2)
        logits = (keys * query).sum(dim=-1) / math.sqrt(keys.shape[-1])
        logits = logits + self.score_bias(pixel_features).squeeze(-1)
        support = valid_cells[:, 0]
        weights = torch.softmax(logits.masked_fill(~support, -torch.inf) / self.temperature, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        selected_depth = (weights * depth_cells[:, 0]).sum(dim=-1, keepdim=False)[:, None]
        output_valid = support.any(dim=-1, keepdim=False)[:, None].to(depth.dtype)
        selected_depth = torch.where(output_valid.bool(), selected_depth, torch.zeros_like(selected_depth))

        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        valid_count = support.sum(dim=-1).clamp_min(1)
        entropy_norm = entropy / valid_count.to(depth.dtype).log().clamp_min(1.0)
        ambiguity = torch.where(output_valid[:, 0].bool(), entropy_norm, torch.ones_like(entropy_norm))[:, None]

        relative_error = (depth_cells[:, 0] - selected_depth[:, 0, :, :, None]).abs()
        dispersion = (weights * relative_error).sum(dim=-1) / selected_depth[:, 0].clamp_min(1e-6)
        dispersion = torch.where(output_valid[:, 0].bool(), dispersion, torch.ones_like(dispersion))[:, None]
        support_fraction = support.to(depth.dtype).mean(dim=-1)[:, None]
        return SurfaceAlignmentOutput(
            depth=selected_depth,
            valid=output_valid,
            ambiguity=ambiguity,
            relative_dispersion=dispersion,
            support_fraction=support_fraction,
            weights=weights[:, None],
        )
