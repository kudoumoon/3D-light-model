import unittest

import torch

from latent_geometry_alignment import align_depth_to_latent
from latent_geometry_head import LatentGeometryHead, points_from_depth
from latent_motion_confidence import LatentMotionConfidence
from latent_reprojection_loss import forward_splat_latent, latent_reprojection_loss


def intrinsics(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(batch, 1, 1)


class Latent3DTests(unittest.TestCase):
    def test_points_from_depth_identity_camera(self):
        depth = torch.ones(1, 1, 2, 2)
        points = points_from_depth(depth, intrinsics())
        self.assertEqual(tuple(points.shape), (1, 3, 2, 2))
        self.assertTrue(torch.allclose(points[:, 2], torch.ones(1, 2, 2)))

    def test_identity_latent_splat_preserves_features(self):
        feature = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        points = points_from_depth(torch.ones(1, 1, 4, 4), intrinsics())
        result = forward_splat_latent(feature, points, torch.ones(1, 1, 4, 4), intrinsics(), torch.eye(4).unsqueeze(0))
        self.assertTrue(torch.all(result.projected_valid))
        self.assertTrue(torch.allclose(result.latent, feature, atol=1e-5))
        metrics = latent_reprojection_loss(result, feature)
        self.assertLess(float(metrics["l1"]), 1e-6)

    def test_alignment_methods_keep_cell_grid(self):
        depth = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4) + 1
        valid = torch.ones_like(depth)
        for method in ("center", "average", "median", "minimum"):
            aligned, support = align_depth_to_latent(depth, valid, (2, 2), method)
            self.assertEqual(tuple(aligned.shape), (1, 1, 2, 2))
            self.assertTrue(torch.all(support))

    def test_heads_preserve_latent_resolution(self):
        latent = torch.randn(2, 16, 4, 5)
        geometry = LatentGeometryHead()(latent, intrinsics(2))
        confidence = LatentMotionConfidence()(latent, geometry.latent_depth, geometry.latent_valid_logits, torch.zeros(2, 6))
        self.assertEqual(tuple(geometry.latent_points.shape), (2, 3, 4, 5))
        self.assertEqual(tuple(confidence.shape), (2, 1, 4, 5))


if __name__ == "__main__":
    unittest.main()
