"""Read-only export contract from latent M1 geometry to Reprojection-WM M2.

The bridge owns no M2 behavior.  It validates and converts one M1 sample into
the NumPy dictionary consumed by M2's existing ``geometry_pose_candidate``.
Extra confidence and provenance fields are preserved even though the current
M2 implementation does not yet consume per-cell motion confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from latent_geometry_head import LatentGeometryOutput


@dataclass(frozen=True)
class M2GeometryPayload:
    geometry: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _single_numpy(value: torch.Tensor, name: str) -> np.ndarray:
    if value.shape[0] != 1:
        raise ValueError(f"{name} export currently requires batch size 1")
    return value.detach().float().cpu().numpy()[0]


def export_latent_geometry_to_m2(
    output: LatentGeometryOutput,
    *,
    valid_threshold: float = 0.5,
) -> M2GeometryPayload:
    """Convert one latent-grid geometry output without changing M2 code."""

    if not 0.0 < valid_threshold < 1.0:
        raise ValueError("valid_threshold must be in (0, 1)")
    depth = _single_numpy(output.latent_depth, "latent_depth")
    points = _single_numpy(output.latent_points, "latent_points")
    valid_probability = _single_numpy(output.latent_valid, "latent_valid")
    intrinsics = _single_numpy(output.intrinsics, "intrinsics")
    if depth.shape[0] != 1 or points.shape[0] != 3 or valid_probability.shape[0] != 1:
        raise ValueError("M1 geometry must contain scalar depth/valid and XYZ points")
    if depth.shape[1:] != points.shape[1:] or depth.shape != valid_probability.shape:
        raise ValueError("depth, points and validity must share the latent grid")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [1, 3, 3]")
    if not np.isfinite(intrinsics).all() or intrinsics[2, 2] == 0:
        raise ValueError("intrinsics must be finite and projective")
    finite_geometry = np.isfinite(points).all(axis=0) & np.isfinite(depth[0])
    positive_depth = depth[0] > 0
    valid = (valid_probability[0] >= valid_threshold) & finite_geometry & positive_depth

    geometry: dict[str, np.ndarray] = {
        "points": np.moveaxis(points, 0, -1).astype(np.float32),
        "depth": depth[0].astype(np.float32),
        "mask": valid,
        "intrinsics": intrinsics.astype(np.float32),
        "source_confidence": valid_probability[0].astype(np.float32),
        "coordinate_convention": np.asarray(
            "opencv_x_right_y_down_z_forward_normalized_intrinsics"
        ),
    }
    confidence_status = "not_attached"
    if output.latent_confidence_logits is not None:
        motion_confidence = _single_numpy(output.latent_confidence, "latent_confidence")
        if motion_confidence.shape != depth.shape:
            raise ValueError("motion confidence must share the latent grid")
        geometry["warp_confidence"] = motion_confidence[0].astype(np.float32)
        confidence_status = "preserved_but_current_m2_geometry_pose_candidate_does_not_consume_it"

    metadata = {
        "contract": "geometry_aligned_latent_3d_to_reprojection_wm_v1",
        "latent_grid_hw": [int(depth.shape[1]), int(depth.shape[2])],
        "spatial_downsample": int(output.spatial_downsample),
        "temporal_downsample": int(output.temporal_downsample),
        "source_coordinate_convention": output.coordinate_convention,
        "export_coordinate_convention": str(geometry["coordinate_convention"]),
        "valid_threshold": float(valid_threshold),
        "valid_fraction": float(valid.mean()),
        "motion_confidence_status": confidence_status,
        "temporal_scope": "single spatial geometry frame; M2 owns chunk horizons",
    }
    return M2GeometryPayload(geometry=geometry, metadata=metadata)
