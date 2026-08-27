import importlib.util
import unittest
import numpy as np

from reproject import _rotation_c2w, forward_splat


def plane():
    h, w = 3, 4
    y, x = np.mgrid[:h, :w]
    points = np.stack([(x + .5) / w - .5, (y + .5) / h - .5, np.ones_like(x)], -1).astype(np.float32)
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    mask = np.ones((h, w), bool)
    k = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]], np.float32)
    return points, rgb, mask, k, np.zeros(3, np.float32), np.eye(3, dtype=np.float32)


class ReprojectionTests(unittest.TestCase):
    def test_identity(self):
        args = plane()
        rgb, depth, mask = forward_splat(*args, 0)
        np.testing.assert_array_equal(rgb, args[1])
        np.testing.assert_array_equal(mask, args[2])
        np.testing.assert_allclose(depth, 1)

    def test_empty_support(self):
        args = list(plane())
        args[2][:] = False
        rgb, depth, mask = forward_splat(*args, 1)
        self.assertFalse(mask.any())
        self.assertFalse(rgb.any())
        self.assertFalse(depth.any())

    def test_points_behind_camera(self):
        args = list(plane())
        args[4][2] = 2
        self.assertFalse(forward_splat(*args, 0)[2].any())

    def test_nearest_depth_wins(self):
        args = list(plane())
        args[2][:] = False
        args[2][0, :2] = True
        args[0][0, 0] = [-.375, -1/3, 1]
        args[0][0, 1] = [-.75, -2/3, 2]
        rgb, depth, mask = forward_splat(*args, 0)
        self.assertEqual(mask.sum(), 1)
        np.testing.assert_array_equal(rgb[0, 0], args[1][0, 0])
        self.assertEqual(depth[0, 0], 1)

    def test_rotation_orthonormal(self):
        r = _rotation_c2w(15, -10)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)

    def test_cuda_identity_when_available(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed; CPU-only validation")
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from reproject_torch import forward_splat_torch
        args = plane()
        result = forward_splat_torch(*(torch.from_numpy(a).cuda() for a in args), 0)
        expected = forward_splat(*args, 0)
        for value, reference in zip(result, expected):
            np.testing.assert_allclose(value.cpu().numpy(), reference)


if __name__ == "__main__":
    unittest.main()
