"""Unit tests for RuleEngine with Pier Stem Reinforcement rules."""

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
            0: "rebar_cage", 1: "vertical_rebar", 2: "stirrup",
            3: "worker", 4: "crane", 5: "pile_cap", 6: "starter_bar",
        },
        "activity": {"name": "Pier Stem Reinforcement", "confidence_threshold": 0.5},
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


class TestRuleEngine(unittest.TestCase):

    def setUp(self):
        rules = [
            {
                "name": "Rebar Cage Present",
                "condition": {"type": "greater_than", "attribute": "rebar_cage_count", "threshold": 0},
                "score": 0.9,
            },
            {
                "name": "Vertical Rebar Installation",
                "condition": {
                    "type": "and",
                    "conditions": [
                        {"type": "greater_than", "attribute": "vertical_rebar_count", "threshold": 2},
                        {"type": "greater_than", "attribute": "worker_count", "threshold": 0},
                    ],
                },
                "score": 0.85,
            },
            {
                "name": "Full Pier Stem Reinforcement",
                "condition": {
                    "type": "and",
                    "conditions": [
                        {"type": "greater_than", "attribute": "vertical_rebar_count", "threshold": 0},
                        {"type": "greater_than", "attribute": "stirrup_count", "threshold": 0},
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

    def test_rebar_cage_rule_matches(self):
        scores = self.engine.apply_rules({"rebar_cage_count": 1})
        self.assertIn("Rebar Cage Present", scores)
        self.assertAlmostEqual(scores["Rebar Cage Present"], 0.9)

    def test_rebar_cage_rule_no_match(self):
        scores = self.engine.apply_rules({"rebar_cage_count": 0})
        self.assertNotIn("Rebar Cage Present", scores)

    def test_vertical_rebar_combined_rule(self):
        attrs = {"vertical_rebar_count": 3, "worker_count": 2}
        scores = self.engine.apply_rules(attrs)
        self.assertIn("Vertical Rebar Installation", scores)

    def test_full_pier_stem_rule(self):
        attrs = {"vertical_rebar_count": 2, "stirrup_count": 3, "worker_count": 1}
        scores = self.engine.apply_rules(attrs)
        self.assertIn("Full Pier Stem Reinforcement", scores)
        self.assertAlmostEqual(scores["Full Pier Stem Reinforcement"], 1.0)

    def test_full_pier_stem_partial_miss(self):
        # Missing stirrup - should NOT fire
        attrs = {"vertical_rebar_count": 2, "stirrup_count": 0, "worker_count": 1}
        scores = self.engine.apply_rules(attrs)
        self.assertNotIn("Full Pier Stem Reinforcement", scores)

    def test_empty_attributes_no_match(self):
        scores = self.engine.apply_rules({})
        self.assertEqual(scores, {})

    def test_multiple_rules_match_simultaneously(self):
        attrs = {
            "rebar_cage_count": 1,
            "vertical_rebar_count": 3,
            "stirrup_count": 2,
            "worker_count": 2,
        }
        scores = self.engine.apply_rules(attrs)
        self.assertIn("Rebar Cage Present", scores)
        self.assertIn("Vertical Rebar Installation", scores)
        self.assertIn("Full Pier Stem Reinforcement", scores)


if __name__ == "__main__":
    unittest.main()
