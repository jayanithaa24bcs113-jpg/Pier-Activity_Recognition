"""
Test Cap Reinforcement Activity

Purpose: Unit tests for CapReinforcementActivityRecognizer (Stage 3).
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Unit tests for CapReinforcementActivityRecognizer (Stage 3).

Covers:
    - "Model Not Trained" short-circuit when model_trained: false
    - "Pier Cap Reinforcement" returned when smoothed score >= threshold
    - "Idle / No Cap Activity" returned when smoothed score < threshold
    - Graceful handling when the detector returns an empty detections list
      (placeholder-safe path — weights file missing on disk)

Two config fixtures are used:
    - ``config_untrained.yaml``   — activities.cap_reinforcement.model_trained: false
    - ``config_trained_no_weights.yaml`` — model_trained: true, but the
      weights file itself does not exist on disk, exercising the
      Detector-level placeholder-safe guard (Step 7) rather than the
      recognizer-level guard (Step 6).
"""

import os

import pytest

from src.activity.activity_recognizer import CapReinforcementActivityRecognizer

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

UNTRAINED_CONFIG = os.path.join(_TEST_DIR, "config_untrained.yaml")
TRAINED_NO_WEIGHTS_CONFIG = os.path.join(_TEST_DIR, "config_trained_no_weight.yaml")


# ---------------------------------------------------------------------
# Scenario 1 — model_trained: false -> "Model Not Trained"
# ---------------------------------------------------------------------

def test_returns_model_not_trained_when_flag_false():
    """When model_trained is False, recognize() must short-circuit and
    return 'Model Not Trained' without touching detector/extractor/rules.
    """
    recognizer = CapReinforcementActivityRecognizer(UNTRAINED_CONFIG)

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
    recognizer = CapReinforcementActivityRecognizer(UNTRAINED_CONFIG)

    # Should not raise FileNotFoundError, AttributeError, or anything else.
    output = recognizer.recognize("/definitely/does/not/exist.jpg")

    assert output["activity"] == "Model Not Trained"


# ---------------------------------------------------------------------
# Scenario 2 — smoothed score >= threshold -> "Pier Cap Reinforcement"
# ---------------------------------------------------------------------

def test_decide_activity_returns_activity_label_when_score_at_threshold():
    """total smoothed score >= 7 -> activity_label."""
    recognizer = CapReinforcementActivityRecognizer(UNTRAINED_CONFIG)
    # model_available is False here, but decide_activity() is pure logic
    # and can be exercised directly without a live detector/model.

    attributes = {"horizontal_rebar_count": 3, "pier_stem_top_count": 1}
    scores = {
        "Horizontal Rebar Present": 3,
        "Pier Stem Top Visible": 3,
        "Pier Cap Reinforcement": 5,
    }  # total = 11 >= 7

    result = recognizer.decide_activity(attributes, scores)

    assert result == "Pier Cap Reinforcement"


def test_decide_activity_returns_activity_label_exactly_at_seven():
    """total smoothed score == 7 (boundary) -> activity_label."""
    recognizer = CapReinforcementActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Horizontal Rebar Present": 3, "Pier Stem Top Visible": 3, "Crane Present": 1}
    # total = 7, exactly at threshold — spec says ">= 7" so this must fire

    result = recognizer.decide_activity({}, scores)

    assert result == "Pier Cap Reinforcement"


# ---------------------------------------------------------------------
# Scenario 3 — smoothed score < threshold -> "Idle / No Cap Activity"
# ---------------------------------------------------------------------

def test_decide_activity_returns_idle_label_when_score_below_threshold():
    """total smoothed score < 7 -> idle_label."""
    recognizer = CapReinforcementActivityRecognizer(UNTRAINED_CONFIG)

    scores = {"Workers Present": 1, "Crane Present": 1}  # total = 2 < 7

    result = recognizer.decide_activity({}, scores)

    assert result == "Idle / No Cap Activity"


def test_decide_activity_returns_idle_label_for_empty_scores():
    """No rules fired at all -> total score 0 -> idle_label."""
    recognizer = CapReinforcementActivityRecognizer(UNTRAINED_CONFIG)

    result = recognizer.decide_activity({}, {})

    assert result == "Idle / No Cap Activity"


# ---------------------------------------------------------------------
# Scenario 4 — detector returns empty list -> graceful handling
# ---------------------------------------------------------------------

def test_recognize_handles_empty_detections_gracefully():
    """When model_trained is True but the weights file is missing on
    disk, Detector.detect() returns [] (placeholder-safe). The full
    recognize() pipeline must still run end-to-end without crashing,
    producing zero-valued attributes and the idle label.
    """
    recognizer = CapReinforcementActivityRecognizer(TRAINED_NO_WEIGHTS_CONFIG)

    # Detector was constructed successfully (model_trained: true), but
    # its underlying YOLO model is None since weights don't exist.
    assert recognizer.model_available is True
    assert recognizer.detector is not None
    assert recognizer.detector.model is None

    output = recognizer.recognize("some/image.jpg")

    # detect() returned [] -> extractor sees no detections -> all zero
    assert output["detections"] == []
    assert output["attributes"]["horizontal_rebar_count"] == 0
    assert output["attributes"]["total_objects"] == 0
    assert output["raw_scores"] == {}
    assert output["activity"] == "Idle / No Cap Activity"


def test_recognize_full_pipeline_does_not_raise():
    """Sanity check: the full recognize() call chain (detect -> extract
    -> rules -> bayesian -> decide) must complete without raising any
    exception when the detector is in placeholder-safe mode.
    """
    recognizer = CapReinforcementActivityRecognizer(TRAINED_NO_WEIGHTS_CONFIG)

    try:
        output = recognizer.recognize("irrelevant/path.jpg")
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"recognize() raised unexpectedly: {exc}")

    assert "activity" in output
    assert "smoothed_scores" in output