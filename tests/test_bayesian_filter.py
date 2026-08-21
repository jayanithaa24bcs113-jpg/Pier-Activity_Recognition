"""Unit tests for BayesianFilter.

These tests validate the smoothing behaviour and edge cases.
"""

import unittest
from src.bayesian.bayesian_filter import BayesianFilter


class TestBayesianFilter(unittest.TestCase):

    def setUp(self):
        self.bf = BayesianFilter(alpha=0.5)

    def test_initial_score_is_raw_value(self):
        result = self.bf.update({"Rebar Density": 0.8})
        self.assertAlmostEqual(result["Rebar Density"], 0.8)

    def test_smoothing_blends_old_and_new(self):
        self.bf.update({"Rebar Density": 1.0})
        result = self.bf.update({"Rebar Density": 0.0})
        # 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(result["Rebar Density"], 0.5)

    def test_multiple_keys(self):
        self.bf.update({"A": 1.0, "B": 0.0})
        result = self.bf.update({"A": 0.0, "B": 1.0})
        self.assertAlmostEqual(result["A"], 0.5)
        self.assertAlmostEqual(result["B"], 0.5)

    def test_reset_clears_state(self):
        self.bf.update({"Rebar Density": 0.8})
        self.bf.reset()
        result = self.bf.update({"Rebar Density": 0.2})
        self.assertAlmostEqual(result["Rebar Density"], 0.2)

    def test_alpha_one_is_no_smoothing(self):
        bf = BayesianFilter(alpha=1.0)
        bf.update({"X": 0.9})
        result = bf.update({"X": 0.1})
        self.assertAlmostEqual(result["X"], 0.1)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            BayesianFilter(alpha=0.0)
        with self.assertRaises(ValueError):
            BayesianFilter(alpha=1.5)


if __name__ == "__main__":
    unittest.main()
