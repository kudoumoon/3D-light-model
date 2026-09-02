"""Chunk-aligned geometry for a causal video-VAE latent grid.

The existing spatial head remains untouched.  This module applies one shared
spatial geometry head to every latent time slice while preserving the causal
Video-VAE time axis explicitly.  Temporal fusion belongs to target construction
and is not hidden inside a fallback branch here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from latent_geometry_head import LatentGeometryHead


def wan_causal_rgb_groups(rgb_frames: int, temporal_compression: int = 4) -> tuple[tuple[int, ...], ...]:
    """Return the exact RGB-frame groups represented by a full Wan latent sequence.

    Wan uses one RGB frame for the first latent and ``temporal_compression``
    causal RGB frames for every later latent.  A complete sequence therefore
    contains ``1 + k * temporal_compression`` RGB frames.
    """

    if temporal_compression <= 0:
        raise ValueError("temporal_compression must be positive")
    if rgb_frames <= 0 or (rgb_frames - 1) % temporal_compression:
        raise ValueError(
            "Wan causal encoding requires rgb_frames = 1 + k * temporal_compression"
        )
    groups: list[tuple[int, ...]] = [(0,)]
    for start in range(1, rgb_frames, temporal_compression):
        groups.append(tuple(range(start, start + temporal_compression)))
    return tuple(groups)


def wan_causal_geometry_anchor_indices(
    rgb_frames: int, temporal_compression: int = 4, anchor_position: int = 2
) -> tuple[int, ...]:
    """Map each Wan latent slice to its audited RGB geometry reference frame.

    Controlled perturbation and independent-frame sweeps identify zero-based
    position 2 (the third frame) as the dominant reference inside each four-frame
    causal group. The singleton first group remains anchored at frame 0.
    """

    if not 0 <= anchor_position < temporal_compression:
        raise ValueError("anchor_position must index a temporal compression group")
    groups = wan_causal_rgb_groups(rgb_frames, temporal_compression)
    return tuple(group[0] if len(group) == 1 else group[anchor_position] for group in groups)


@dataclass(frozen=True)
class ChunkLatentGeometryOutput:
    """Geometry attached one-to-one to ``[B,C,F,H,W]`` Video-VAE latents."""

    latent_depth: torch.Tensor
    latent_points: torch.Tensor
    latent_valid_logits: torch.Tensor
    intrinsics: torch.Tensor
    spatial_downsample: int
    temporal_downsample: int
    temporal_anchor_position: int = 2
    coordinate_convention: str = "camera_xyz_normalized_intrinsics"

    @property
    def latent_valid(self) -> torch.Tensor:
        return torch.sigmoid(self.latent_valid_logits)


def _expand_chunk_intrinsics(
    intrinsics: torch.Tensor, batch: int, frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if intrinsics.ndim == 3 and intrinsics.shape == (batch, 3, 3):
        chunk = intrinsics[:, None].expand(-1, frames, -1, -1)
    elif intrinsics.ndim == 4 and intrinsics.shape == (batch, frames, 3, 3):
        chunk = intrinsics
    else:
        raise ValueError("intrinsics must be [B,3,3] or [B,F,3,3]")
    return chunk, chunk.reshape(batch * frames, 3, 3)


class ChunkLatentGeometryHead(nn.Module):
    """Shared lightweight geometry head over causal latent time slices.

    The model deliberately performs no temporal averaging.  Each latent slice
    receives the geometry target defined by a separately audited causal target
    constructor, preventing an arbitrary RGB-frame anchor from being baked into
    the network architecture.
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
        self.spatial = LatentGeometryHead(
            latent_channels=latent_channels,
            width=width,
            blocks=blocks,
            spatial_downsample=spatial_downsample,
            temporal_downsample=temporal_downsample,
        )

    def forward(
        self, latent: torch.Tensor, intrinsics: torch.Tensor
    ) -> ChunkLatentGeometryOutput:
        if latent.ndim != 5:
            raise ValueError("latent must have shape [B,C,F,H,W]")
        batch, channels, frames, height, width = latent.shape
        chunk_intrinsics, flat_intrinsics = _expand_chunk_intrinsics(
            intrinsics, batch, frames
        )
        flat_latent = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        flat = self.spatial(flat_latent, flat_intrinsics)
        depth = flat.latent_depth.reshape(batch, frames, 1, height, width).permute(
            0, 2, 1, 3, 4
        )
        points = flat.latent_points.reshape(batch, frames, 3, height, width).permute(
            0, 2, 1, 3, 4
        )
        valid_logits = flat.latent_valid_logits.reshape(
            batch, frames, 1, height, width
        ).permute(0, 2, 1, 3, 4)
        return ChunkLatentGeometryOutput(
            latent_depth=depth,
            latent_points=points,
            latent_valid_logits=valid_logits,
            intrinsics=chunk_intrinsics,
            spatial_downsample=self.spatial.spatial_downsample,
            temporal_downsample=self.spatial.temporal_downsample,
        )
