import unittest

import torch

from latent_chunk_geometry import (
    ChunkLatentGeometryHead,
    wan_causal_geometry_anchor_indices,
    wan_causal_rgb_groups,
)
from latent_surface_alignment import ReprojectionOptimalSurfaceSelector


class ChunkGeometryTests(unittest.TestCase):
    def test_wan_causal_groups_are_explicit(self):
        self.assertEqual(
            wan_causal_rgb_groups(13),
            ((0,), (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
        )
        with self.assertRaises(ValueError):
            wan_causal_rgb_groups(12)

    def test_wan_geometry_anchor_uses_audited_third_frame(self):
        self.assertEqual(
            wan_causal_geometry_anchor_indices(13),
            (0, 3, 7, 11),
        )
        with self.assertRaises(ValueError):
            wan_causal_geometry_anchor_indices(13, anchor_position=4)

    def test_chunk_head_preserves_time_and_grid(self):
        latent = torch.randn(2, 16, 3, 4, 5, requires_grad=True)
        intrinsics = torch.tensor(
            [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]
        ).repeat(2, 1, 1)
        output = ChunkLatentGeometryHead(width=16, blocks=1)(latent, intrinsics)
        self.assertEqual(tuple(output.latent_depth.shape), (2, 1, 3, 4, 5))
        self.assertEqual(tuple(output.latent_points.shape), (2, 3, 3, 4, 5))
        self.assertEqual(tuple(output.latent_valid.shape), (2, 1, 3, 4, 5))
        self.assertEqual(output.temporal_anchor_position, 2)
        output.latent_depth.mean().backward()
        self.assertIsNotNone(latent.grad)

    def test_surface_selector_exposes_ambiguity(self):
        torch.manual_seed(7)
        depth = torch.tensor(
            [[[[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 8.0, 8.0], [2.0, 2.0, 8.0, 8.0]]]]
        )
        valid = torch.ones_like(depth)
        confidence = torch.ones_like(depth)
        latent = torch.randn(1, 4, 2, 2)
        selector = ReprojectionOptimalSurfaceSelector(latent_channels=4, hidden=8)
        output = selector(depth, valid, latent, confidence)
        self.assertEqual(tuple(output.depth.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(output.weights.shape), (1, 1, 2, 2, 4))
        self.assertTrue(torch.allclose(output.weights.sum(dim=-1), torch.ones(1, 1, 2, 2)))
        self.assertTrue(torch.isfinite(output.ambiguity).all())
        self.assertTrue(torch.isfinite(output.relative_dispersion).all())


if __name__ == "__main__":
    unittest.main()
