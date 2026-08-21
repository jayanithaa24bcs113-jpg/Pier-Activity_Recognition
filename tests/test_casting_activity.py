"""End-to-end tests for CastingActivityRecognizer.decide_activity.

These tests exercise the decision logic directly (attributes + smoothed
scores - activity label) without invoking YOLO, keeping tests fast and
deterministic, consistent with the unit-level style used elsewhere in
this test suite.
"""

import unittest
from unittest.mock import MagicMock

from src.activity.activity_recognizer import CastingActivityRecognizer


def _make_recognizer_stub() -> CastingActivityRecognizer:
    """Build a CastingActivityRecognizer with components mocked out.

    Bypasses ``_build_components`` (which would load YOLO weights and
    real config) so only ``decide_activity`` is under test.

    Returns:
        CastingActivityRecognizer: Instance with labels set directly.
    """
    recognizer = CastingActivityRecognizer.__new__(CastingActivityRecognizer)
    recognizer.activity_label = "Pier Stem Casting"
    recognizer.idle_label = "Idle / No Casting Activity"
    recognizer.score_threshold = 0.5
    return recognizer


class TestCastingActivityRecognizer(unittest.TestCase):

    def setUp(self):
        self.recognizer = _make_recognizer_stub()

    def test_full_pier_stem_casting_shortcut(self):
        attrs = {"formwork_count": 1, "concrete_pump_count": 1}
        scores = {"Full Pier Stem Casting": 1.0}
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Pier Stem Casting")

    def test_concrete_placement_active_shortcut(self):
        attrs = {"formwork_count": 0, "concrete_pump_count": 1}
        scores = {"Concrete Placement Active": 0.8}
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Pier Stem Casting")

    def test_total_score_threshold_with_formwork_present(self):
        attrs = {"formwork_count": 1, "concrete_pump_count": 0}
        scores = {
            "Formwork Installed": 0.6,
            "Worker Near Formwork": 0.4,
            "Vibrator Active": 0.4,
        }
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Pier Stem Casting")

    def test_total_score_threshold_without_minimum_presence_fails(self):
        # total score >= 1.2 but no formwork or concrete pump detected
        attrs = {"formwork_count": 0, "concrete_pump_count": 0}
        scores = {
            "Worker Near Concrete": 0.4,
            "High Concrete Density": 0.4,
            "Casted Pier Visible": 0.4,
        }
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Idle / No Casting Activity")

    def test_low_total_score_returns_idle(self):
        attrs = {"formwork_count": 1, "concrete_pump_count": 0}
        scores = {"Formwork Installed": 0.6}
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Idle / No Casting Activity")

    def test_empty_scores_returns_idle(self):
        attrs = {"formwork_count": 0, "concrete_pump_count": 0}
        scores = {}
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Idle / No Casting Activity")

    def test_concrete_pump_alone_satisfies_minimum_presence(self):
        attrs = {"formwork_count": 0, "concrete_pump_count": 1}
        scores = {
            "Concrete Pump Active": 0.4,
            "Transit Mixer Present": 0.4,
            "Vibrator Active": 0.4,
        }
        activity = self.recognizer.decide_activity(attrs, scores)
        self.assertEqual(activity, "Pier Stem Casting")


if __name__ == "__main__":
    unittest.main()