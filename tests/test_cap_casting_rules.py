"""
Test Cap Casting Rules

Purpose: Unit tests for rules_cap_casting.yaml via RuleEngine (Stage 4).
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Unit tests for rules_cap_casting.yaml via RuleEngine (Stage 4).

Constructs a real ``RuleEngine`` pointed at
``src/rules/rules_cap_casting.yaml`` and evaluates it against
hand-built attribute dictionaries, verifying each rule fires (and does
not fire) as expected, plus both composite rules and total score
accumulation.
"""

import pytest

from src.rules.rule_engine import RuleEngine

RULES_FILE = "src/rules/rules_cap_casting.yaml"


def zero_attributes() -> dict:
    """Return an attribute dict with every Stage 4 attribute at its
    zero/False baseline value.

    Returns:
        dict: All-zero attribute dictionary matching the shape produced
            by ``CapCastingAttributeExtractor.extract()``.
    """
    return {
        "cap_formwork_count": 0,
        "concrete_pump_count": 0,
        "transit_mixer_count": 0,
        "vibrator_count": 0,
        "casted_cap_count": 0,
        "pier_stem_count": 0,
        "worker_count": 0,
        "concrete_density": 0.0,
        "worker_near_formwork": False,
        "worker_near_casted_cap": False,
        "total_objects": 0,
    }


@pytest.fixture
def engine() -> RuleEngine:
    """Build a RuleEngine loaded with the Stage 4 rules file.

    Returns:
        RuleEngine: Engine configured with rules_cap_casting.yaml.
    """
    return RuleEngine(rules_file=RULES_FILE)


# ---------------------------------------------------------------------
# Individual rule tests
# ---------------------------------------------------------------------

def test_cap_formwork_installed_fires(engine):
    attrs = zero_attributes()
    attrs["cap_formwork_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Cap Formwork Installed") == 0.6


def test_pier_stem_visible_fires(engine):
    attrs = zero_attributes()
    attrs["pier_stem_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Pier Stem Visible") == 0.6


def test_worker_near_formwork_fires(engine):
    attrs = zero_attributes()
    attrs["worker_near_formwork"] = True

    scores = engine.apply_rules(attrs)

    assert scores.get("Worker Near Formwork") == 0.4


def test_worker_near_formwork_does_not_fire_when_false(engine):
    attrs = zero_attributes()
    attrs["worker_near_formwork"] = False

    scores = engine.apply_rules(attrs)

    assert "Worker Near Formwork" not in scores


def test_transit_mixer_present_fires(engine):
    attrs = zero_attributes()
    attrs["transit_mixer_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Transit Mixer Present") == 0.4


def test_concrete_pump_active_fires(engine):
    attrs = zero_attributes()
    attrs["concrete_pump_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Concrete Pump Active") == 0.4


def test_vibrator_active_fires(engine):
    attrs = zero_attributes()
    attrs["vibrator_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Vibrator Active") == 0.4


def test_casted_cap_visible_fires(engine):
    attrs = zero_attributes()
    attrs["casted_cap_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Casted Cap Visible") == 0.6


def test_worker_near_casted_cap_fires(engine):
    attrs = zero_attributes()
    attrs["worker_near_casted_cap"] = True

    scores = engine.apply_rules(attrs)

    assert scores.get("Worker Near Casted Cap") == 0.4


def test_high_concrete_density_fires(engine):
    attrs = zero_attributes()
    attrs["concrete_density"] = 0.31

    scores = engine.apply_rules(attrs)

    assert scores.get("High Concrete Density") == 0.4


def test_high_concrete_density_does_not_fire_at_threshold(engine):
    """concrete_density == 0.3 should NOT fire (condition is strictly > 0.3)."""
    attrs = zero_attributes()
    attrs["concrete_density"] = 0.3

    scores = engine.apply_rules(attrs)

    assert "High Concrete Density" not in scores


# ---------------------------------------------------------------------
# Composite rules
# ---------------------------------------------------------------------

def test_concrete_placement_active_fires_when_both_present(engine):
    """Concrete Placement Active: pump > 0 AND transit_mixer > 0."""
    attrs = zero_attributes()
    attrs["concrete_pump_count"] = 1
    attrs["transit_mixer_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Concrete Placement Active") == 0.8


def test_concrete_placement_active_does_not_fire_with_only_pump(engine):
    attrs = zero_attributes()
    attrs["concrete_pump_count"] = 1
    # transit_mixer_count left at 0

    scores = engine.apply_rules(attrs)

    assert "Concrete Placement Active" not in scores


def test_full_pier_cap_casting_fires_with_pump(engine):
    """Full Pier Cap Casting: formwork>0 AND (pump>0 OR mixer>0) AND worker>0."""
    attrs = zero_attributes()
    attrs["cap_formwork_count"] = 1
    attrs["concrete_pump_count"] = 1
    attrs["worker_count"] = 1

    scores = engine.apply_rules(attrs)

    assert scores.get("Full Pier Cap Casting") == 1.0


def test_full_pier_cap_casting_fires_with_mixer_instead_of_pump(engine):
    """The OR branch should also fire via transit_mixer alone (no pump)."""
    attrs = zero_attributes()
    attrs["cap_formwork_count"] = 1
    attrs["transit_mixer_count"] = 1
    attrs["worker_count"] = 1
    # concrete_pump_count left at 0

    scores = engine.apply_rules(attrs)

    assert scores.get("Full Pier Cap Casting") == 1.0


def test_full_pier_cap_casting_does_not_fire_without_worker(engine):
    """Missing the worker>0 leg of the AND must prevent the rule from firing."""
    attrs = zero_attributes()
    attrs["cap_formwork_count"] = 1
    attrs["concrete_pump_count"] = 1
    # worker_count left at 0

    scores = engine.apply_rules(attrs)

    assert "Full Pier Cap Casting" not in scores


def test_full_pier_cap_casting_does_not_fire_without_formwork(engine):
    """Missing formwork>0 must prevent the rule even if pump/mixer/worker present."""
    attrs = zero_attributes()
    attrs["concrete_pump_count"] = 1
    attrs["transit_mixer_count"] = 1
    attrs["worker_count"] = 1
    # cap_formwork_count left at 0

    scores = engine.apply_rules(attrs)

    assert "Full Pier Cap Casting" not in scores


def test_full_pier_cap_casting_does_not_fire_when_no_branch_satisfied(engine):
    """Formwork and worker present, but neither pump nor mixer -> OR fails."""
    attrs = zero_attributes()
    attrs["cap_formwork_count"] = 1
    attrs["worker_count"] = 1
    # both concrete_pump_count and transit_mixer_count left at 0

    scores = engine.apply_rules(attrs)

    assert "Full Pier Cap Casting" not in scores


# ---------------------------------------------------------------------
# Total score accumulation
# ---------------------------------------------------------------------

def test_total_score_accumulates_correctly(engine):
    """Multiple simultaneously-true rules should sum to the expected total."""
    attrs = zero_attributes()
    attrs["cap_formwork_count"] = 1     # Cap Formwork Installed (0.6)
    attrs["pier_stem_count"] = 1        # Pier Stem Visible (0.6)
    attrs["concrete_pump_count"] = 1    # Concrete Pump Active (0.4)
    attrs["transit_mixer_count"] = 1    # Transit Mixer Present (0.4)
    attrs["worker_count"] = 1           # (no standalone worker rule)
    # Also satisfies: Concrete Placement Active (0.8),
    # Full Pier Cap Casting (1.0)

    scores = engine.apply_rules(attrs)
    total = sum(scores.values())

    expected_total = 0.6 + 0.6 + 0.4 + 0.4 + 0.8 + 1.0  # = 3.8
    assert total == pytest.approx(expected_total, abs=0.0001)


def test_total_score_all_rules_fire(engine):
    """A saturated attribute dict should fire every rule and sum correctly."""
    attrs = {
        "cap_formwork_count": 2,
        "concrete_pump_count": 1,
        "transit_mixer_count": 1,
        "vibrator_count": 1,
        "casted_cap_count": 1,
        "pier_stem_count": 1,
        "worker_count": 2,
        "concrete_density": 0.5,
        "worker_near_formwork": True,
        "worker_near_casted_cap": True,
        "total_objects": 9,
    }

    scores = engine.apply_rules(attrs)
    total = sum(scores.values())

    # 0.6+0.6+0.4+0.4+0.4+0.4+0.6+0.4+0.4+0.8+1.0 = all 11 rules fire
    assert len(scores) == 11
    assert total == pytest.approx(6.0, abs=0.0001)


# ---------------------------------------------------------------------
# All-zero attributes -> zero score
# ---------------------------------------------------------------------

def test_all_zero_attributes_returns_zero_score(engine):
    attrs = zero_attributes()

    scores = engine.apply_rules(attrs)

    assert scores == {}
    assert sum(scores.values()) == 0