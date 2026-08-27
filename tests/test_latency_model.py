import unittest

from latency_model import interval_cost, lookup_latency


class LatencyTests(unittest.TestCase):
    rows = [{"active_ratio": 1, "median_ms": 40},
            {"active_ratio": .5, "median_ms": 24},
            {"active_ratio": .125, "median_ms": 10}]

    def test_measured_points(self):
        for row in self.rows:
            self.assertEqual(lookup_latency(self.rows, row["active_ratio"]), row["median_ms"])

    def test_floor_is_not_interpolation_to_zero(self):
        self.assertEqual(lookup_latency(self.rows, .01), 10)
        self.assertAlmostEqual(lookup_latency(self.rows, .01, "legacy_linear"), .8)

    def test_zero_is_explicit_bypass(self):
        self.assertEqual(lookup_latency(self.rows, 0), 0)

    def test_inside_sample_range(self):
        self.assertEqual(lookup_latency(self.rows, .75), 32)

    def test_interval_one(self):
        result = interval_cost(self.rows, .8, geometry_ms=20, warp_ms=3, interval=1)
        self.assertEqual(result["baseline_ms_proxy"], 160)
        self.assertEqual(result["average_ms_proxy"], 180)

    def test_invalid_ratio(self):
        for ratio in [-1, 1.01, float("nan")]:
            with self.assertRaises(ValueError):
                lookup_latency(self.rows, ratio)


if __name__ == "__main__":
    unittest.main()
