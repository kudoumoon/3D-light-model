import unittest
import numpy as np

from quality_metrics import tile_candidate_oracle_fraction, tile_pass_fraction


class QualityTests(unittest.TestCase):
    def test_pixel_oracle_is_not_tile_choice(self):
        copy = np.array([[0., 1.], [0., 1.]])
        warp = 1 - copy
        mask = np.ones((2, 2), bool)
        self.assertEqual(tile_pass_fraction(np.minimum(copy, warp), mask, .1, 2), 1)
        self.assertEqual(tile_candidate_oracle_fraction(copy, warp, mask, .1, 2), 0)

    def test_copy_always_available(self):
        c = np.zeros((3, 3))
        mask = np.zeros((3, 3), bool)
        self.assertEqual(tile_candidate_oracle_fraction(c, c, mask, .1, 2), 1)
        self.assertEqual(tile_pass_fraction(c, mask, .1, 2), 0)

    def test_partial_tile_counts_once(self):
        e = np.zeros((3, 3))
        e[2, 2] = 1
        self.assertEqual(tile_pass_fraction(e, np.ones((3, 3), bool), .1, 2), .75)

    def test_support_threshold(self):
        mask = np.ones((10, 10), bool)
        mask[0, :6] = False
        zeros = np.zeros((10, 10))
        self.assertEqual(tile_pass_fraction(zeros, mask, 0, 10), 0)
        mask[0, 5] = True
        self.assertEqual(tile_pass_fraction(zeros, mask, 0, 10), 1)

    def test_invalid_tile(self):
        with self.assertRaises(ValueError):
            tile_pass_fraction(np.zeros((2, 2)), np.ones((2, 2), bool), .1, 0)


if __name__ == "__main__":
    unittest.main()
