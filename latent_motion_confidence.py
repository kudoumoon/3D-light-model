"""Motion-conditioned confidence for frozen latent geometry."""

from __future__ import annotations

import torch
from torch import nn


class LatentMotionConfidence(nn.Module):
    """Predict per-cell reuse confidence without changing geometry weights.

    ``motion`` is a normalized vector, normally yaw, pitch and translation in
    the source-camera coordinate system.  Geometry is passed as detached input
    by the training loop after the latent geometry head has converged.
    """

    def __init__(self, latent_channels: int = 16, motion_dim: int = 6, width: int = 48) -> None:
        super().__init__()
        if min(latent_channels, motion_dim, width) <= 0:
            raise ValueError("all channel dimensions must be positive")
        groups = 8 if width % 8 == 0 else 1
        self.motion_dim = motion_dim
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels + 2 + motion_dim, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def forward(
        self, latent: torch.Tensor, depth: torch.Tensor, valid_logits: torch.Tensor, motion: torch.Tensor
    ) -> torch.Tensor:
        if latent.ndim != 4 or depth.shape[1] != 1 or valid_logits.shape[1] != 1:
            raise ValueError("latent/depth/valid tensors must be BCHW with scalar geometry maps")
        if motion.shape != (latent.shape[0], self.motion_dim):
            raise ValueError(f"motion must have shape [B, {self.motion_dim}]")
        motion_map = motion.to(latent).view(latent.shape[0], self.motion_dim, 1, 1)
        motion_map = motion_map.expand(-1, -1, latent.shape[-2], latent.shape[-1])
        return self.net(torch.cat((latent, depth, valid_logits, motion_map), dim=1))
