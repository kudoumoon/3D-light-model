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
    # Binary occupancy on the target grid. This is deliberately distinct
    # from accumulated splat mass: one exact source-to-target cell match must
    # count as fully covered, not as 1-exp(-1).
    coverage: torch.Tensor
    projected_valid: torch.Tensor
    support_mass: torch.Tensor
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
    occupancy_threshold: float = 1e-3,
    eps: float = 1e-6,
) -> LatentWarpResult:
    """Project feature cells to the target latent grid with soft z-buffering."""

    _validate(points, features, intrinsics, source_to_target)
    if valid.shape != (points.shape[0], 1, points.shape[2], points.shape[3]):
        raise ValueError("valid must be [B, 1, H, W]")
    if depth_temperature <= 0:
        raise ValueError("depth_temperature must be positive")
    if not 0 < occupancy_threshold <= 1:
        raise ValueError("occupancy_threshold must be in (0, 1]")
    # Scatter accumulation is carried out in fp32 even when Wan latents are
    # bf16. Geometry weights and z-buffer terms would otherwise create a
    # mixed-dtype scatter operation.
    features = features.float()
    points = points.float()
    valid = valid.float()
    intrinsics = intrinsics.float()
    source_to_target = source_to_target.float()
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
    # Invalid geometry has zero support, but scatter still requires every
    # temporary index to be legal. Replace only its indexing coordinates.
    u_index = torch.nan_to_num(u, nan=0.5, posinf=1.0, neginf=0.0)
    v_index = torch.nan_to_num(v, nan=0.5, posinf=1.0, neginf=0.0)
    x = (u_index.clamp(0, 1) * width - 0.5).clamp(0, width - 1)
    y = (v_index.clamp(0, 1) * height - 0.5).clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=width - 1), (y0 + 1).clamp(max=height - 1)
    wx, wy = x - x0.to(x), y - y0.to(y)
    numerator = torch.zeros(batch, channels, height * width, device=features.device, dtype=features.dtype)
    denominator = torch.zeros(batch, 1, height * width, device=features.device, dtype=features.dtype)
    depth_sum = torch.zeros_like(denominator)
    contributions = (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    )

    # Determine the nearest surface independently for every target cell.
    # A batch-global median is not a z-buffer and can suppress valid distant
    # surfaces even when no collision exists.
    nearest_depth = torch.full_like(denominator, torch.inf)
    for xi, yi, bilinear in contributions:
        index = yi * width + xi
        active = (support > 0) & (bilinear > 0)
        # Visibility selection is discrete. Detaching only this arg-min pass
        # avoids an in-place scatter-reduce autograd conflict while gradients
        # still flow through projection coordinates and the second-pass gate.
        candidate = torch.where(active, z.detach(), torch.full_like(z, torch.inf)).unsqueeze(1)
        nearest_depth.scatter_reduce_(2, index.unsqueeze(1), candidate, reduce="amin", include_self=True)

    flat_features = features.flatten(2)
    for xi, yi, bilinear in contributions:
        index = yi * width + xi
        local_nearest = nearest_depth.gather(2, index.unsqueeze(1)).squeeze(1)
        relative_gap = (z - local_nearest) / local_nearest.clamp_min(eps)
        relative_gap = torch.nan_to_num(relative_gap, nan=0.0, posinf=20.0, neginf=0.0).clamp_min(0)
        depth_gate = torch.exp((-depth_temperature * relative_gap).clamp(min=-20.0, max=0.0))
        weight = (support * bilinear * depth_gate).unsqueeze(1)
        numerator.scatter_add_(2, index.unsqueeze(1).expand(-1, channels, -1), flat_features * weight)
        denominator.scatter_add_(2, index.unsqueeze(1), weight)
        depth_sum.scatter_add_(2, index.unsqueeze(1), weight * z.unsqueeze(1))
    projected_valid = denominator >= occupancy_threshold
    coverage = projected_valid.to(features.dtype)
    warped = numerator / denominator.clamp_min(eps)
    return LatentWarpResult(
        latent=warped.view(batch, channels, height, width),
        coverage=coverage.view(batch, 1, height, width),
        projected_valid=projected_valid.view(batch, 1, height, width),
        support_mass=denominator.view(batch, 1, height, width),
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
    mask = warped.projected_valid.to(target_latent.dtype)
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
    return {
        "loss": total,
        "l1": l1,
        "cosine": cosine,
        "coverage": mask.mean(),
        "hole_ratio": 1.0 - mask.mean(),
        "support_mass_mean": warped.support_mass.mean(),
    }


def compare_warp_to_copy(
    warped: LatentWarpResult,
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compare Warp and Copy on exactly the same target support.

    Full-frame metrics are also returned. The composite output is an M1-only
    proxy: it uses warped features where projection is valid and the source
    latent in holes. It never reads target features to fill a hole.
    """

    if warped.latent.shape != source_latent.shape or source_latent.shape != target_latent.shape:
        raise ValueError("warped, source and target latents must have identical shapes")
    mask = warped.projected_valid.to(target_latent.dtype)
    channels = target_latent.shape[1]
    normalizer = (mask.sum() * channels).clamp_min(eps)

    warp_valid_l1 = ((warped.latent - target_latent).abs() * mask).sum() / normalizer
    copy_valid_l1 = ((source_latent - target_latent).abs() * mask).sum() / normalizer
    warp_cosine = F.cosine_similarity(warped.latent, target_latent, dim=1, eps=eps)
    copy_cosine = F.cosine_similarity(source_latent, target_latent, dim=1, eps=eps)
    mask_2d = mask[:, 0]
    mask_count = mask_2d.sum().clamp_min(eps)
    composite = torch.where(warped.projected_valid, warped.latent, source_latent)

    return {
        "warp_valid_l1": warp_valid_l1,
        "copy_valid_l1": copy_valid_l1,
        "warp_valid_cosine_similarity": (warp_cosine * mask_2d).sum() / mask_count,
        "copy_valid_cosine_similarity": (copy_cosine * mask_2d).sum() / mask_count,
        "warp_full_l1": (warped.latent - target_latent).abs().mean(),
        "copy_full_l1": (source_latent - target_latent).abs().mean(),
        "composite_full_l1": (composite - target_latent).abs().mean(),
        "coverage": mask.mean(),
        "hole_ratio": 1.0 - mask.mean(),
        "composite": composite,
    }
