"""Unit tests for RuleEngine using rules_stem_casting.yaml.

Mirrors the fixture style of test_rule_engine.py: a temporary config
file points to a temporary rules file so tests are fully isolated and
do not depend on the real project config.
"""

import os
import unittest
import tempfile
import yaml

from src.rules.rule_engine import RuleEngine


def _make_config(rules_path: str) -> str:
    config = {
        "model": {"weights": "yolov8s.pt", "save_dir": "models"},
        "dataset": {
            "root": "datasets",
            "train_path": "datasets/train",
            "valid_path": "datasets/valid",
            "test_path": "datasets/test",
        },
        "training": {"epochs": 10, "imgsz": 640, "device": "0"},
        "metrics": {"output_file": "outputs/metrics.csv"},
        "rules": {"file": rules_path},
        "classes": {
            0: "Formwork", 1: "Concrete Pump", 2: "Transit Mixer",
            3: "Vibrator", 4: "Fresh Concrete", 5: "Worker", 6: "Casted Pier",
        },
        "activity": {"name": "Pier Stem Casting", "confidence_threshold": 0.5},
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp)
    tmp.close()
    return tmp.name


def _make_rules(rules: list) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(rules, tmp)
    tmp.close()
    return tmp.name


class TestCastingRuleEngine(unittest.TestCase):

    def setUp(self):
        rules = [
            {
                "name": "Formwork Installed",
                "condition": {"type": "greater_than", "attribute": "formwork_count", "threshold": 0},
                "score": 0.6,
            },
            {
                "name": "Worker Near Formwork",
                "condition": {"type": "equals", "attribute": "worker_near_formwork", "threshold": True},
                "score": 0.4,
            },
            {
                "name": "Fresh Concrete Visible",
                "condition": {"type": "greater_than", "attribute": "fresh_concrete_count", "threshold": 0},
                "score": 0.6,
            },
            {
                "name": "High Concrete Density",
                "condition": {"type": "greater_than", "attribute": "concrete_density", "threshold": 0.3},
                "score": 0.4,
            },
            {
                "name": "Tall Vertical Shuttering",
                "condition": {"type": "greater_than", "attribute": "tall_vertical_shuttering", "threshold": 0.5},
                "score": 0.6,
            },
            {
                "name": "Concrete Placement Active",
                "condition": {
                    "type": "and",
                    "conditions": [
                        {"type": "greater_than", "attribute": "concrete_pump_count", "threshold": 0},
                        {"type": "greater_than", "attribute": "fresh_concrete_count", "threshold": 0},
                    ],
                },
                "score": 0.8,
            },
            {
                "name": "Full Pier Stem Casting",
                "condition": {
                    "type": "and",
                    "conditions": [
                        {"type": "greater_than", "attribute": "formwork_count", "threshold": 0},
                        {
                            "type": "or",
                            "conditions": [
                                {"type": "greater_than", "attribute": "concrete_pump_count", "threshold": 0},
                                {"type": "greater_than", "attribute": "transit_mixer_count", "threshold": 0},
                            ],
                        },
                        {"type": "greater_than", "attribute": "worker_count", "threshold": 0},
                    ],
                },
                "score": 1.0,
            },
        ]
        self.rules_path = _make_rules(rules)
        self.config_path = _make_config(self.rules_path)
        self.engine = RuleEngine(self.config_path)

    def tearDown(self):
        os.unlink(self.rules_path)
        os.unlink(self.config_path)

    def test_formwork_rule_matches(self):
        scores = self.engine.apply_rules({"formwork_count": 1})
        self.assertIn("Formwork Installed", scores)
        self.assertEqual(scores["Formwork Installed"], 0.6)

    def test_formwork_rule_no_match(self):
        scores = self.engine.apply_rules({"formwork_count": 0})
        self.assertNotIn("Formwork Installed", scores)

    def test_worker_near_formwork_bool_match(self):
        scores = self.engine.apply_rules({"worker_near_formwork": True})
        self.assertIn("Worker Near Formwork", scores)

    def test_worker_near_formwork_bool_no_match(self):
        scores = self.engine.apply_rules({"worker_near_formwork": False})
        self.assertNotIn("Worker Near Formwork", scores)

    def test_high_concrete_density_threshold(self):
        scores = self.engine.apply_rules({"concrete_density": 0.5})
        self.assertIn("High Concrete Density", scores)

    def test_high_concrete_density_below_threshold(self):
        scores = self.engine.apply_rules({"concrete_density": 0.1})
        self.assertNotIn("High Concrete Density", scores)

    def test_tall_vertical_shuttering_weighted_by_confidence(self):
        scores = self.engine.apply_rules(
            {"tall_vertical_shuttering": 0.8},
            obj_conf=0.5,
        )
        self.assertIn("Tall Vertical Shuttering", scores)
        # base score 0.6 * obj_conf 0.5 = 0.3
        self.assertAlmostEqual(scores["Tall Vertical Shuttering"], 0.3)

    def test_tall_vertical_shuttering_default_conf_is_full_score(self):
        scores = self.engine.apply_rules({"tall_vertical_shuttering": 0.8})
        self.assertAlmostEqual(scores["Tall Vertical Shuttering"], 0.6)

    def test_concrete_placement_active_requires_both(self):
        attrs = {"concrete_pump_count": 1, "fresh_concrete_count": 1}
        scores = self.engine.apply_rules(attrs)
        self.assertIn("Concrete Placement Active", scores)

    def test_concrete_placement_active_partial_miss(self):
        attrs = {"concrete_pump_count": 1, "fresh_concrete_count": 0}
        scores = self.engine.apply_rules(attrs)
        self.assertNotIn("Concrete Placement Active", scores)

    def test_full_pier_stem_casting_with_concrete_pump(self):
        attrs = {
            "formwork_count": 1,
            "concrete_pump_count": 1,
            "transit_mixer_count": 0,
            "worker_count": 1,
        }
        scores = self.engine.apply_rules(attrs)
        self.assertIn("Full Pier Stem Casting", scores)
        self.assertEqual(scores["Full Pier Stem Casting"], 1.0)

    def test_full_pier_stem_casting_with_transit_mixer_instead(self):
        attrs = {
            "formwork_count": 1,
            "concrete_pump_count": 0,
            "transit_mixer_count": 1,
            "worker_count": 1,
        }
        scores = self.engine.apply_rules(attrs)
        self.assertIn("Full Pier Stem Casting", scores)

    def test_full_pier_stem_casting_missing_worker_fails(self):
        attrs = {
            "formwork_count": 1,
            "concrete_pump_count": 1,
            "transit_mixer_count": 0,
            "worker_count": 0,
        }
        scores = self.engine.apply_rules(attrs)
        self.assertNotIn("Full Pier Stem Casting", scores)

    def test_empty_attributes_no_match(self):
        scores = self.engine.apply_rules({})
        self.assertEqual(scores, {})


if __name__ == "__main__":
    unittest.main()