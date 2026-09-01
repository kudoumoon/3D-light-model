import unittest

import numpy as np
import torch

from latent_geometry_head import LatentGeometryOutput, points_from_depth
from latent_m2_bridge import export_latent_geometry_to_m2


def make_output(*, attach_confidence: bool = True) -> LatentGeometryOutput:
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]
    )
    depth = torch.ones(1, 1, 4, 5)
    output = LatentGeometryOutput(
        latent_depth=depth,
        latent_points=points_from_depth(depth, intrinsics),
        latent_valid_logits=torch.full_like(depth, 20.0),
        latent_confidence_logits=None,
        intrinsics=intrinsics,
        spatial_downsample=8,
        temporal_downsample=4,
    )
    return output.with_confidence(torch.zeros_like(depth)) if attach_confidence else output


class LatentM2BridgeTests(unittest.TestCase):
    def test_exports_exact_m2_spatial_contract(self):
        payload = export_latent_geometry_to_m2(make_output())
        geometry = payload.geometry
        self.assertEqual(geometry["points"].shape, (4, 5, 3))
        self.assertEqual(geometry["depth"].shape, (4, 5))
        self.assertEqual(geometry["mask"].shape, (4, 5))
        self.assertEqual(geometry["intrinsics"].shape, (3, 3))
        self.assertEqual(geometry["warp_confidence"].shape, (4, 5))
        self.assertTrue(geometry["mask"].all())
        self.assertTrue(np.isfinite(geometry["points"]).all())
        self.assertEqual(payload.metadata["latent_grid_hw"], [4, 5])

    def test_unattached_confidence_remains_explicit(self):
        payload = export_latent_geometry_to_m2(make_output(attach_confidence=False))
        self.assertNotIn("warp_confidence", payload.geometry)
        self.assertEqual(payload.metadata["motion_confidence_status"], "not_attached")

    def test_rejects_silent_batch_collapse(self):
        output = make_output()
        doubled = LatentGeometryOutput(
            latent_depth=output.latent_depth.repeat(2, 1, 1, 1),
            latent_points=output.latent_points.repeat(2, 1, 1, 1),
            latent_valid_logits=output.latent_valid_logits.repeat(2, 1, 1, 1),
            latent_confidence_logits=None,
            intrinsics=output.intrinsics.repeat(2, 1, 1),
            spatial_downsample=8,
            temporal_downsample=4,
        )
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            export_latent_geometry_to_m2(doubled)


if __name__ == "__main__":
    unittest.main()
