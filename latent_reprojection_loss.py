"""Latent-grid 3D splatting and reprojection losses.

The renderer is intentionally small and explicit.  It uses bilinear forward
splatting and a soft depth gate, exposing rather than hiding disocclusion and
many-to-one conflicts through its coverage map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentWarpResult:
    latent: torch.Tensor
    coverage: torch.Tensor
    projected_valid: torch.Tensor
    target_depth: torch.Tensor


def _validate(points: torch.Tensor, features: torch.Tensor, intrinsics: torch.Tensor, transform: torch.Tensor) -> None:
    if points.ndim != 4 or points.shape[1] != 3:
        raise ValueError("points must be [B, 3, H, W]")
    if features.ndim != 4 or features.shape[0] != points.shape[0] or features.shape[-2:] != points.shape[-2:]:
        raise ValueError("features must align with points on the latent grid")
    if intrinsics.shape != (points.shape[0], 3, 3):
        raise ValueError("intrinsics must be [B, 3, 3]")
    if transform.shape != (points.shape[0], 4, 4):
        raise ValueError("transform must be [B, 4, 4] source-camera to target-camera")


def forward_splat_latent(
    features: torch.Tensor,
    points: torch.Tensor,
    valid: torch.Tensor,
    intrinsics: torch.Tensor,
    source_to_target: torch.Tensor,
    depth_temperature: float = 8.0,
    eps: float = 1e-6,
) -> LatentWarpResult:
    """Project feature cells to the target latent grid with soft z-buffering."""

    _validate(points, features, intrinsics, source_to_target)
    if valid.shape != (points.shape[0], 1, points.shape[2], points.shape[3]):
        raise ValueError("valid must be [B, 1, H, W]")
    if depth_temperature <= 0:
        raise ValueError("depth_temperature must be positive")
    batch, channels, height, width = features.shape
    xyz = points.flatten(2).transpose(1, 2)
    rotation = source_to_target[:, :3, :3]
    translation = source_to_target[:, :3, 3]
    target = xyz @ rotation.transpose(1, 2) + translation[:, None, :]
    z = target[..., 2]
    u = intrinsics[:, 0, 0:1] * (target[..., 0] / z.clamp_min(eps)) + intrinsics[:, 0, 2:3]
    v = intrinsics[:, 1, 1:2] * (target[..., 1] / z.clamp_min(eps)) + intrinsics[:, 1, 2:3]
    inside = (z > eps) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    support = valid.flatten(1).clamp(0, 1) * inside.to(features.dtype)
    # Normalized coordinates refer to cell centres: u=(x+0.5)/W.
    x = u.clamp(0, 1) * width - 0.5
    y = v.clamp(0, 1) * height - 0.5
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=width - 1), (y0 + 1).clamp(max=height - 1)
    wx, wy = x - x0.to(x), y - y0.to(y)
    # Relative depth preserves unit mass for a surface at the batch median.
    # Absolute-depth exponentials would make an identity warp look like a hole.
    relative_depth = z / z.detach().median(dim=1, keepdim=True).values.clamp_min(eps)
    depth_gate = torch.exp((-depth_temperature * (relative_depth - 1.0)).clamp(max=20.0))
    numerator = torch.zeros(batch, channels, height * width, device=features.device, dtype=features.dtype)
    denominator = torch.zeros(batch, 1, height * width, device=features.device, dtype=features.dtype)
    depth_sum = torch.zeros_like(denominator)
    flat_features = features.flatten(2)
    for xi, yi, bilinear in ((x0, y0, (1 - wx) * (1 - wy)), (x1, y0, wx * (1 - wy)), (x0, y1, (1 - wx) * wy), (x1, y1, wx * wy)):
        index = yi * width + xi
        weight = (support * bilinear * depth_gate).unsqueeze(1)
        numerator.scatter_add_(2, index.unsqueeze(1).expand(-1, channels, -1), flat_features * weight)
        denominator.scatter_add_(2, index.unsqueeze(1), weight)
        depth_sum.scatter_add_(2, index.unsqueeze(1), weight * z.unsqueeze(1))
    coverage = 1.0 - torch.exp(-denominator)
    warped = numerator / denominator.clamp_min(eps)
    return LatentWarpResult(
        latent=warped.view(batch, channels, height, width),
        coverage=coverage.view(batch, 1, height, width),
        projected_valid=coverage.view(batch, 1, height, width) > 1e-3,
        target_depth=(depth_sum / denominator.clamp_min(eps)).view(batch, 1, height, width),
    )


def latent_reprojection_loss(
    warped: LatentWarpResult,
    target_latent: torch.Tensor,
    mode: str = "l1_cosine",
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return coverage-masked feature losses; holes are reported separately."""

    if warped.latent.shape != target_latent.shape:
        raise ValueError("warped and target latents must have identical shapes")
    mask = warped.coverage.clamp(0, 1)
    l1 = ((warped.latent - target_latent).abs() * mask).sum() / (mask.sum() * target_latent.shape[1]).clamp_min(eps)
    cosine = 1.0 - F.cosine_similarity(warped.latent, target_latent, dim=1, eps=eps)
    cosine = (cosine.unsqueeze(1) * mask).sum() / mask.sum().clamp_min(eps)
    if mode == "l1":
        total = l1
    elif mode == "cosine":
        total = cosine
    elif mode == "l1_cosine":
        total = l1 + cosine
    else:
        raise ValueError("mode must be l1, cosine or l1_cosine")
    return {"loss": total, "l1": l1, "cosine": cosine, "coverage": mask.mean(), "hole_ratio": 1.0 - mask.mean()}
