"""Geometry-Aligned Latent 3D head v2.

The public contract is identical to :mod:`latent_geometry_head`.  Internally
it adds camera rays, multi-scale context, a geometry/context split, and
cell-level ambiguity prediction.  The auxiliary predictions are intentionally
kept internal so downstream M2 shape contracts remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from latent_geometry_head import LatentGeometryOutput, points_from_depth


class _Residual(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class LatentGeometryHeadV2(nn.Module):
    """Ray-conditioned, uncertainty-aware latent geometry head."""

    def __init__(self, latent_channels: int = 16, width: int = 64, blocks: int = 3,
                 spatial_downsample: int = 8, temporal_downsample: int = 4) -> None:
        super().__init__()
        if min(latent_channels, width, blocks) <= 0:
            raise ValueError("latent_channels, width and blocks must be positive")
        self.stem = nn.Conv2d(latent_channels + 2, width, 3, padding=1)
        self.local = nn.Sequential(*[_Residual(width, 1) for _ in range(blocks)])
        self.global_context = nn.Sequential(_Residual(width, 2), _Residual(width, 4))
        self.geometry_branch = nn.Sequential(_Residual(width), _Residual(width))
        self.context_branch = nn.Sequential(_Residual(width), _Residual(width, 2))
        self.geometry = nn.Conv2d(width, 2, 1)
        self.auxiliary = nn.Conv2d(width, 3, 1)
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample

    @staticmethod
    def _rays(intrinsics: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            (torch.arange(height, device=intrinsics.device, dtype=dtype) + .5) / height,
            (torch.arange(width, device=intrinsics.device, dtype=dtype) + .5) / width,
            indexing="ij",
        )
        fx, fy = intrinsics[:, 0, 0].view(-1, 1, 1), intrinsics[:, 1, 1].view(-1, 1, 1)
        cx, cy = intrinsics[:, 0, 2].view(-1, 1, 1), intrinsics[:, 1, 2].view(-1, 1, 1)
        return torch.stack(((x - cx) / fx, (y - cy) / fy), dim=1)

    def forward(self, latent: torch.Tensor, intrinsics: torch.Tensor) -> LatentGeometryOutput:
        if latent.ndim != 4 or intrinsics.shape != (latent.shape[0], 3, 3):
            raise ValueError("latent must be [B,C,H,W] and intrinsics must be [B,3,3]")
        rays = self._rays(intrinsics, latent.shape[-2], latent.shape[-1], latent.dtype)
        x = F.silu(self.stem(torch.cat((latent, rays), dim=1)))
        x = self.local(x)
        context = F.interpolate(self.global_context(F.avg_pool2d(x, 2)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        geometry_feature = self.geometry_branch(x + context)
        context_feature = self.context_branch(x + context)
        prediction = self.geometry(geometry_feature)
        aux = self.auxiliary(context_feature)
        log_depth = prediction[:, :1]
        depth = torch.exp(log_depth.clamp(-8.0, 8.0))
        return LatentGeometryOutput(
            latent_depth=depth,
            latent_points=points_from_depth(depth, intrinsics),
            latent_valid_logits=prediction[:, 1:2],
            latent_confidence_logits=None,
            intrinsics=intrinsics,
            spatial_downsample=self.spatial_downsample,
            temporal_downsample=self.temporal_downsample,
        )

    def forward_with_auxiliary(self, latent: torch.Tensor, intrinsics: torch.Tensor) -> tuple[LatentGeometryOutput, dict[str, torch.Tensor]]:
        """Return fixed-contract geometry plus internal uncertainty/edge signals."""
        if latent.ndim != 4 or intrinsics.shape != (latent.shape[0], 3, 3):
            raise ValueError("latent must be [B,C,H,W] and intrinsics must be [B,3,3]")
        rays = self._rays(intrinsics, latent.shape[-2], latent.shape[-1], latent.dtype)
        x = F.silu(self.stem(torch.cat((latent, rays), dim=1)))
        x = self.local(x)
        context = F.interpolate(self.global_context(F.avg_pool2d(x, 2)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        geometry_feature = self.geometry_branch(x + context)
        context_feature = self.context_branch(x + context)
        prediction, aux = self.geometry(geometry_feature), self.auxiliary(context_feature)
        depth = torch.exp(prediction[:, :1].clamp(-8.0, 8.0))
        output = LatentGeometryOutput(depth, points_from_depth(depth, intrinsics), prediction[:, 1:2], None, intrinsics, self.spatial_downsample, self.temporal_downsample)
        return output, {"depth_log_variance": aux[:, :1], "boundary_logits": aux[:, 1:2], "surface_separation": F.softplus(aux[:, 2:3])}
