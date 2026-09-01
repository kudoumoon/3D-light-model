"""Lightweight geometry head aligned with a frozen world-model latent grid.

The module deliberately predicts depth, validity and confidence rather than an
unconstrained XYZ map.  Points are reconstructed analytically from normalized
camera intrinsics, keeping the output compatible with projective warping.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


CoordinateConvention = Literal["camera_xyz_normalized_intrinsics"]


@dataclass(frozen=True)
class LatentGeometryOutput:
    """Geometry attached one-to-one to a spatial latent grid.

    Tensor layouts are ``[B, C, H_latent, W_latent]``.  ``intrinsics`` uses
    normalized image coordinates, so ``K[0, 2]`` and ``K[1, 2]`` are in [0, 1].
    """

    latent_depth: torch.Tensor
    latent_points: torch.Tensor
    latent_valid_logits: torch.Tensor
    latent_confidence_logits: torch.Tensor | None
    intrinsics: torch.Tensor
    spatial_downsample: int
    temporal_downsample: int
    coordinate_convention: CoordinateConvention = "camera_xyz_normalized_intrinsics"

    @property
    def latent_valid(self) -> torch.Tensor:
        return torch.sigmoid(self.latent_valid_logits)

    @property
    def latent_confidence(self) -> torch.Tensor:
        if self.latent_confidence_logits is None:
            raise RuntimeError(
                "motion-conditioned confidence is not attached; run the frozen-geometry confidence head first"
            )
        return torch.sigmoid(self.latent_confidence_logits)

    def with_confidence(self, logits: torch.Tensor) -> "LatentGeometryOutput":
        expected = (self.latent_depth.shape[0], 1, *self.latent_depth.shape[-2:])
        if logits.shape != expected:
            raise ValueError(f"confidence logits must have shape {expected}")
        return replace(self, latent_confidence_logits=logits)


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


def points_from_depth(
    depth: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Back-project latent-cell centres using normalized intrinsics.

    The spatial grid is defined by latent-cell centres, not by resizing an RGB
    point map.  This is the analytic part of Geometry-Aligned Latent 3D.
    """

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B, 1, H, W]")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, 3, 3]")
    batch, _, height, width = depth.shape
    if intrinsics.shape[0] not in (1, batch):
        raise ValueError("intrinsics batch size must be 1 or match depth")
    if intrinsics.shape[0] == 1 and batch != 1:
        intrinsics = intrinsics.expand(batch, -1, -1)
    y, x = torch.meshgrid(
        (torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5) / height,
        (torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5) / width,
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0].view(batch, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1)
    z = depth[:, 0]
    return torch.stack(((x - cx) / fx * z, (y - cy) / fy * z, z), dim=1)


class LatentGeometryHead(nn.Module):
    """A compact head operating directly on frozen VAE latents.

    It has no RGB encoder and does not alter the VAE.  A single head predicts
    only latent-grid depth and valid support. Motion-conditioned confidence is
    attached only after geometry training through an independent frozen-geometry head.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        width: int = 64,
        blocks: int = 3,
        spatial_downsample: int = 8,
        temporal_downsample: int = 4,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or width <= 0 or blocks <= 0:
            raise ValueError("latent_channels, width and blocks must be positive")
        self.stem = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[_Block(width) for _ in range(blocks)])
        self.geometry = nn.Conv2d(width, 2, 1)
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample

    def forward(
        self, latent: torch.Tensor, intrinsics: torch.Tensor
    ) -> LatentGeometryOutput:
        if latent.ndim != 4:
            raise ValueError("latent must have shape [B, C, H, W]")
        feature = self.blocks(F.silu(self.stem(latent)))
        geometry = self.geometry(feature)
        depth = F.softplus(geometry[:, :1]) + 1e-4
        valid_logits = geometry[:, 1:2]
        return LatentGeometryOutput(
            latent_depth=depth,
            latent_points=points_from_depth(depth, intrinsics),
            latent_valid_logits=valid_logits,
            latent_confidence_logits=None,
            intrinsics=intrinsics,
            spatial_downsample=self.spatial_downsample,
            temporal_downsample=self.temporal_downsample,
        )
