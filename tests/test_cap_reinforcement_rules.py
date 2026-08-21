"""
Test Cap Reinforcement Rules

Purpose: Unit tests for rules_cap_reinforcement.yaml via RuleEngine (Stage 3).
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Unit tests for rules_cap_reinforcement.yaml via RuleEngine (Stage 3).

Constructs a real ``RuleEngine`` pointed at
``src/rules/rules_cap_reinforcement.yaml`` and evaluates it against
hand-built attribute dictionaries, verifying each rule fires (and does
not fire) as expected.
"""

import pytest

from src.rules.rule_engine import RuleEngine

RULES_FILE = "src/rules/rules_cap_reinforcement.yaml"


# ---------------------------------------------------------------------
# Baseline all-zero attribute dict — every test starts from this and
# overrides only the fields relevant to the rule under test, so each
# test is explicit about exactly which attribute(s) trigger the rule.
# ---------------------------------------------------------------------

def zero_attributes() -> dict:
    """Return an attribute dict with every Stage 3 attribute at its
    zero/False baseline value.

    Returns:
        dict: All-zero attribute dictionary matching the shape produced
            by ``CapReinforcementAttributeExtractor.extract()``.
    """
    return {
        "horizontal_rebar_count": 0,
        "pier_stem_top_count": 0,
        "rebar_cage_count": 0,
        "worker_count": 0,
        "crane_count": 0,
        "rebar_density": 0.0,
        "avg_rebar_aspect_ratio": 0.0,
        "avg_cage_aspect_ratio": 0.0,
        "worker_near_rebar": False,
        "total_objects": 0,
    }


@pytest.fixture
def engine() -> RuleEngine:
    """Build a RuleEngine loaded with the Stage 3 rules file.

    Returns:
        RuleEngine: Engine configured with rules_cap_reinforcement.yaml.
    """
    return RuleEngine(rules_file=RULES_FILE)


# ---------------------------------------------------------------------
# Individual rule tests — each fires correctly with a matching
# attribute dict, and does not fire otherwise.
# ---------------------------------------------------------------------

def test_horizontal_rebar_present_fires(engine):
    attrs = zero_attributes()
    attrs["horizontal_rebar_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Horizontal Rebar Present") == 3


def test_pier_stem_top_visible_fires(engine):
    attrs = zero_attributes()
    attrs["pier_stem_top_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Pier Stem Top Visible") == 3


def test_rebar_cage_present_fires(engine):
    attrs = zero_attributes()
    attrs["rebar_cage_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Rebar Cage Present") == 2


def test_workers_present_fires(engine):
    attrs = zero_attributes()
    attrs["worker_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Workers Present") == 1


def test_crane_present_fires(engine):
    attrs = zero_attributes()
    attrs["crane_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Crane Present") == 1


def test_dense_cap_reinforcement_fires(engine):
    attrs = zero_attributes()
    attrs["rebar_density"] = 0.31

    scores = engine.apply_rules(attrs)

    assert scores.get("Dense Cap Reinforcement") == 2


def test_dense_cap_reinforcement_does_not_fire_at_threshold(engine):
    """rebar_density == 0.30 should NOT fire (condition is strictly > 0.30)."""
    attrs = zero_attributes()
    attrs["rebar_density"] = 0.30

    scores = engine.apply_rules(attrs)

    assert "Dense Cap Reinforcement" not in scores


def test_wide_rebar_geometry_confirmed_fires(engine):
    attrs = zero_attributes()
    attrs["avg_rebar_aspect_ratio"] = 4.5

    scores = engine.apply_rules(attrs)

    assert scores.get("Wide Rebar Geometry Confirmed") == 2


def test_worker_near_rebar_fires(engine):
    attrs = zero_attributes()
    attrs["worker_near_rebar"] = True

    scores = engine.apply_rules(attrs)

    assert scores.get("Worker Near Rebar") == 2


def test_worker_near_rebar_does_not_fire_when_false(engine):
    attrs = zero_attributes()
    attrs["worker_near_rebar"] = False

    scores = engine.apply_rules(attrs)

    assert "Worker Near Rebar" not in scores


# ---------------------------------------------------------------------
# Composite Rule 9 — each OR branch fires independently
# ---------------------------------------------------------------------

def test_composite_rule_branch_1_fires(engine):
    """Branch 1: horizontal_rebar_count > 2 AND pier_stem_top_count > 0."""
    attrs = zero_attributes()
    attrs["horizontal_rebar_count"] = 3
    attrs["pier_stem_top_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Pier Cap Reinforcement") == 5


def test_composite_rule_branch_1_does_not_fire_alone(engine):
    """horizontal_rebar_count > 2 alone (no pier_stem_top) must not fire branch 1."""
    attrs = zero_attributes()
    attrs["horizontal_rebar_count"] = 3
    # pier_stem_top_count left at 0

    scores = engine.apply_rules(attrs)

    assert "Pier Cap Reinforcement" not in scores


def test_composite_rule_branch_2_fires(engine):
    """Branch 2: rebar_cage_count > 0 AND rebar_density > 0.30."""
    attrs = zero_attributes()
    attrs["rebar_cage_count"] = 1
    attrs["rebar_density"] = 0.35

    scores = engine.apply_rules(attrs)

    assert scores.get("Pier Cap Reinforcement") == 5


def test_composite_rule_branch_3_fires(engine):
    """Branch 3: horizontal_rebar_count > 0 AND avg_rebar_aspect_ratio > 4.0."""
    attrs = zero_attributes()
    attrs["horizontal_rebar_count"] = 1
    attrs["avg_rebar_aspect_ratio"] = 5.0

    scores = engine.apply_rules(attrs)

    assert scores.get("Pier Cap Reinforcement") == 5


def test_composite_rule_no_branch_satisfied(engine):
    """None of the three OR branches satisfied -> composite rule must not fire."""
    attrs = zero_attributes()
    attrs["horizontal_rebar_count"] = 1  # branch 1 needs >2, branch 3 needs aspect>4.0
    attrs["rebar_cage_count"] = 0        # branch 2 needs cage>0

    scores = engine.apply_rules(attrs)

    assert "Pier Cap Reinforcement" not in scores


# ---------------------------------------------------------------------
# Total score accumulation
# ---------------------------------------------------------------------

def test_total_score_accumulates_correctly(engine):
    """Multiple simultaneously-true rules should sum to the expected total."""
    attrs = zero_attributes()
    attrs["horizontal_rebar_count"] = 3      # Horizontal Rebar Present (3)
    attrs["pier_stem_top_count"] = 1         # Pier Stem Top Visible (3)
    attrs["worker_count"] = 1                # Workers Present (1)
    # Also satisfies composite branch 1 -> Pier Cap Reinforcement (5)

    scores = engine.apply_rules(attrs)
    total = sum(scores.values())

    expected_total = 3 + 3 + 1 + 5  # = 12
    assert total == expected_total
    assert total >= 7  # crosses the activity threshold


def test_total_score_all_rules_fire(engine):
    """A saturated attribute dict should fire every rule and sum correctly."""
    attrs = {
        "horizontal_rebar_count": 5,
        "pier_stem_top_count": 2,
        "rebar_cage_count": 2,
        "worker_count": 3,
        "crane_count": 1,
        "rebar_density": 0.5,
        "avg_rebar_aspect_ratio": 6.0,
        "avg_cage_aspect_ratio": 3.0,
        "worker_near_rebar": True,
        "total_objects": 13,
    }

    scores = engine.apply_rules(attrs)
    total = sum(scores.values())

    # 3+3+2+1+1+2+2+2+5 = 21 (all 9 rules fire)
    assert len(scores) == 9
    assert total == 21


# ---------------------------------------------------------------------
# All-zero attributes -> zero score
# ---------------------------------------------------------------------

def test_all_zero_attributes_returns_zero_score(engine):
    attrs = zero_attributes()

    scores = engine.apply_rules(attrs)

    assert scores == {}
    assert sum(scores.values()) == 0