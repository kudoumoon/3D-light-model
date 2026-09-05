import unittest

import torch

from latent_geometry_alignment import align_depth_to_latent
from latent_geometry_head import LatentGeometryHead, points_from_depth
from latent_motion_confidence import LatentMotionConfidence
from latent_reprojection_loss import LatentWarpResult, compare_warp_to_copy, forward_splat_latent, latent_reprojection_loss, merge_latent_warps_priority


def intrinsics(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]).repeat(batch, 1, 1)


class Latent3DTests(unittest.TestCase):
    def test_priority_merge_only_fills_holes(self):
        first_valid = torch.tensor([[[[True, False, True]]]])
        second_valid = torch.tensor([[[[True, True, False]]]])
        first = LatentWarpResult(torch.tensor([[[[1.0, 0.0, 3.0]]]]), first_valid.float(), first_valid, first_valid.float(), torch.ones(1, 1, 1, 3))
        second = LatentWarpResult(torch.tensor([[[[9.0, 2.0, 0.0]]]]), second_valid.float(), second_valid, second_valid.float() * 2, torch.ones(1, 1, 1, 3) * 2)
        merged = merge_latent_warps_priority((first, second))
        self.assertTrue(torch.equal(merged.projected_valid, torch.tensor([[[[True, True, True]]]])))
        self.assertTrue(torch.equal(merged.latent, torch.tensor([[[[1.0, 2.0, 3.0]]]])))
        self.assertEqual(float(merged.support_mass[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(merged.support_mass[0, 0, 0, 1]), 2.0)

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
        self.assertTrue(torch.allclose(result.coverage, torch.ones_like(result.coverage)))
        self.assertTrue(torch.allclose(result.support_mass, torch.ones_like(result.support_mass)))
        self.assertTrue(torch.allclose(result.latent, feature, atol=1e-5))
        metrics = latent_reprojection_loss(result, feature)
        self.assertLess(float(metrics["l1"]), 1e-6)
        self.assertAlmostEqual(float(metrics["coverage"]), 1.0, places=6)

    def test_one_cell_translation_has_exact_occupancy(self):
        feature = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
        points = points_from_depth(torch.ones(1, 1, 1, 4), intrinsics())
        transform = torch.eye(4).unsqueeze(0)
        transform[:, 0, 3] = 0.25
        result = forward_splat_latent(feature, points, torch.ones(1, 1, 1, 4), intrinsics(), transform)
        self.assertTrue(torch.equal(result.projected_valid.flatten(), torch.tensor([False, True, True, True])))
        self.assertTrue(torch.allclose(result.latent.flatten()[1:], feature.flatten()[:3], atol=1e-5))
        self.assertAlmostEqual(float(result.coverage.mean()), 0.75, places=6)

    def test_local_z_buffer_prefers_near_surface(self):
        feature = torch.tensor([[[[10.0, 100.0]]]])
        points = torch.tensor([[[[-0.25, -0.50]], [[0.0, 0.0]], [[1.0, 2.0]]]])
        result = forward_splat_latent(
            feature,
            points,
            torch.ones(1, 1, 1, 2),
            intrinsics(),
            torch.eye(4).unsqueeze(0),
            depth_temperature=12.0,
        )
        self.assertTrue(result.projected_valid[0, 0, 0, 0])
        self.assertFalse(result.projected_valid[0, 0, 0, 1])
        self.assertLess(float((result.latent[0, 0, 0, 0] - 10.0).abs()), 0.01)

    def test_warp_copy_comparison_uses_identical_mask(self):
        source = torch.tensor([[[[0.0, 4.0, 3.0, 2.0]]]])
        target = torch.tensor([[[[9.0, 0.0, 4.0, 3.0]]]])
        points = points_from_depth(torch.ones(1, 1, 1, 4), intrinsics())
        transform = torch.eye(4).unsqueeze(0)
        transform[:, 0, 3] = 0.25
        result = forward_splat_latent(source, points, torch.ones(1, 1, 1, 4), intrinsics(), transform)
        comparison = compare_warp_to_copy(result, source, target)
        self.assertLess(float(comparison["warp_valid_l1"]), 1e-6)
        self.assertGreater(float(comparison["copy_valid_l1"]), 0.0)
        self.assertTrue(torch.equal(comparison["composite"], torch.tensor([[[[0.0, 0.0, 4.0, 3.0]]]])))

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
        with self.assertRaises(RuntimeError):
            _ = geometry.latent_confidence
        attached = geometry.with_confidence(confidence)
        self.assertTrue(torch.allclose(attached.latent_confidence, torch.sigmoid(confidence)))
        self.assertIsNone(geometry.latent_confidence_logits)
        self.assertEqual(tuple(geometry.latent_points.shape), (2, 3, 4, 5))
        self.assertEqual(tuple(confidence.shape), (2, 1, 4, 5))


if __name__ == "__main__":
    unittest.main()
