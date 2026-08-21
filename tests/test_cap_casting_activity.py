"""
Test Cap Casting Activity

Purpose: Unit tests for CapCastingActivityRecognizer (Stage 4).
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Unit tests for CapCastingActivityRecognizer (Stage 4).

Covers:
    - "Model Not Trained" short-circuit when model_trained: false
    - "Pier Cap Casting" returned when the composite/shortcut rules fire
    - "Idle / No Cap Casting Activity" returned when no shortcut fires
    - Graceful handling when the detector returns an empty detections
      list (placeholder-safe path — weights file missing on disk)

Two config fixtures are used, resolved relative to this test file's own
directory (not the pytest working directory) so the tests are robust
regardless of where pytest is invoked from:
    - ``config_cap_casting_untrained.yaml`` — model_trained: false
    - ``config_cap_casting_trained_no_weights.yaml`` — model_trained:
      true, but the weights file itself does not exist on disk,
      exercising the Detector-level placeholder-safe guard (Step 7)
      rather than the recognizer-level guard (Step 6).
"""

import os

import pytest

from src.activity.activity_recognizer import CapCastingActivityRecognizer

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

UNTRAINED_CONFIG = os.path.join(_TEST_DIR, "config_cap_casting_untrained.yaml")
TRAINED_NO_WEIGHTS_CONFIG = os.path.join(
    _TEST_DIR, "config_cap_casting_trained_no_weights.yaml"
)


# ---------------------------------------------------------------------
# Scenario 1 — model_trained: false -> "Model Not Trained"
# ---------------------------------------------------------------------

def test_returns_model_not_trained_when_flag_false():
    """When model_trained is False, recognize() must short-circuit and
    return 'Model Not Trained' without touching detector/extractor/rules.
    """
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    assert recognizer.model_available is False
    assert recognizer.detector is None
    assert recognizer.extractor is None
    assert recognizer.rule_engine is None

    output = recognizer.recognize("some/nonexistent/image.jpg")

    assert output["activity"] == "Model Not Trained"
    assert output["attributes"] == {}
    assert output["raw_scores"] == {}
    assert output["smoothed_scores"] == {}
    assert output["detections"] == []


def test_model_not_trained_does_not_raise_for_missing_image():
    """Even a genuinely nonexistent image path must not raise, since the
    guard short-circuits before any file I/O or model call is attempted.
    """
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    output = recognizer.recognize("/definitely/does/not/exist.jpg")

    assert output["activity"] == "Model Not Trained"


# ---------------------------------------------------------------------
# Scenario 2 — shortcut/threshold rules -> "Pier Cap Casting"
# ---------------------------------------------------------------------

def test_decide_activity_fires_via_full_pier_cap_casting_shortcut():
    """'Full Pier Cap Casting' score >= 0.8 -> activity_label, regardless
    of total_score or minimum_presence.
    """
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Full Pier Cap Casting": 1.0}
    attributes = {"cap_formwork_count": 0, "concrete_pump_count": 0}

    result = recognizer.decide_activity(attributes, scores)

    assert result == "Pier Cap Casting"


def test_decide_activity_fires_via_concrete_placement_active_shortcut():
    """'Concrete Placement Active' score >= 0.6 -> activity_label."""
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Concrete Placement Active": 0.8}
    attributes = {}

    result = recognizer.decide_activity(attributes, scores)

    assert result == "Pier Cap Casting"


def test_decide_activity_fires_via_general_threshold():
    """total_score >= 1.2 AND minimum_presence -> activity_label."""
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Cap Formwork Installed": 0.6, "Pier Stem Visible": 0.6}  # total = 1.2
    attributes = {"cap_formwork_count": 1, "concrete_pump_count": 0}

    result = recognizer.decide_activity(attributes, scores)

    assert result == "Pier Cap Casting"


def test_decide_activity_general_threshold_requires_minimum_presence():
    """total_score >= 1.2 alone is NOT enough without cap_formwork or pump."""
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Worker Near Formwork": 0.4, "Vibrator Active": 0.4, "Casted Cap Visible": 0.6}
    # total = 1.4 >= 1.2, but neither cap_formwork_count nor concrete_pump_count present
    attributes = {"cap_formwork_count": 0, "concrete_pump_count": 0}

    result = recognizer.decide_activity(attributes, scores)

    assert result == "Idle / No Cap Casting Activity"


# ---------------------------------------------------------------------
# Scenario 3 — no shortcut fires -> "Idle / No Cap Casting Activity"
# ---------------------------------------------------------------------

def test_decide_activity_returns_idle_label_for_empty_scores():
    """No rules fired at all -> idle_label."""
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    result = recognizer.decide_activity({}, {})

    assert result == "Idle / No Cap Casting Activity"


def test_decide_activity_returns_idle_label_for_low_total_score():
    """Total score below 1.2 -> idle_label even with minimum presence."""
    recognizer = CapCastingActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Vibrator Active": 0.4}  # total = 0.4
    attributes = {"cap_formwork_count": 1, "concrete_pump_count": 0}

    result = recognizer.decide_activity(attributes, scores)

    assert result == "Idle / No Cap Casting Activity"


# ---------------------------------------------------------------------
# Scenario 4 — detector returns empty list -> graceful handling
# ---------------------------------------------------------------------

def test_recognize_handles_empty_detections_gracefully():
    """When model_trained is True but the weights file is missing on
    disk, Detector.detect() returns [] (placeholder-safe). The full
    recognize() pipeline must still run end-to-end without crashing,
    producing zero-valued attributes and the idle label.
    """
    recognizer = CapCastingActivityRecognizer(TRAINED_NO_WEIGHTS_CONFIG)

    assert recognizer.model_available is True
    assert recognizer.detector is not None
    assert recognizer.detector.model is None

    output = recognizer.recognize("some/image.jpg")

    assert output["detections"] == []
    assert output["attributes"]["cap_formwork_count"] == 0
    assert output["attributes"]["total_objects"] == 0
    assert output["raw_scores"] == {}
    assert output["activity"] == "Idle / No Cap Casting Activity"


def test_recognize_full_pipeline_does_not_raise():
    """Sanity check: the full recognize() call chain (detect -> extract
    -> rules -> bayesian -> decide) must complete without raising any
    exception when the detector is in placeholder-safe mode.
    """
    recognizer = CapCastingActivityRecognizer(TRAINED_NO_WEIGHTS_CONFIG)

    try:
        output = recognizer.recognize("irrelevant/path.jpg")
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"recognize() raised unexpectedly: {exc}")

    assert "activity" in output
    assert "smoothed_scores" in output