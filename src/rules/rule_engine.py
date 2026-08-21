"""
Rule Engine

Purpose: Rule Engine for pier construction activity scoring.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Rule Engine for pier construction activity scoring.

Rules are defined in a YAML file and describe conditions based on attributes
extracted from detection results.  Each matching rule contributes a numeric
score that downstream components (e.g. BayesianFilter) can consume.

Supports multiple activities by accepting a ``rules_file`` path at
initialisation.  When no ``rules_file`` is supplied the path is read from
``config['rules']['file']`` preserving full backward compatibility.

The ``apply_rules`` method accepts an optional ``obj_conf`` value used to
weight the ``Tall Vertical Shuttering`` rule score by object confidence,
as specified in the Pier Stem Casting rule logic.
"""

import os
import yaml
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Condition evaluators
# ---------------------------------------------------------------------------

def _evaluate_condition(condition: dict, attributes: dict) -> bool:
    """Recursively evaluate a single condition dictionary.

    Supported condition types:
        - ``greater_than``: attribute value > threshold
        - ``less_than``:    attribute value < threshold
        - ``equals``:       attribute value == threshold
        - ``and``:          all nested conditions must be True
        - ``or``:           at least one nested condition must be True

    Args:
        condition: Condition specification dict from rules YAML.
        attributes: Attribute dict extracted from detections.

    Returns:
        bool: Whether the condition is satisfied.
    """
    ctype = condition.get("type")

    if ctype == "and":
        return all(
            _evaluate_condition(c, attributes)
            for c in condition.get("conditions", [])
        )

    if ctype == "or":
        return any(
            _evaluate_condition(c, attributes)
            for c in condition.get("conditions", [])
        )

    attr_name = condition.get("attribute")
    threshold = condition.get("threshold", 0)
    value = attributes.get(attr_name, 0)

    if ctype == "greater_than":
        return value > threshold
    if ctype == "less_than":
        return value < threshold
    if ctype == "equals":
        return value == threshold

    logger.warning(f"Unknown condition type: {ctype}")
    return False


# ---------------------------------------------------------------------------
# RuleEngine class
# ---------------------------------------------------------------------------

class RuleEngine:
    """Evaluate attribute dictionaries against configured rules.

    Args:
        config_path: Path to the main YAML configuration file.
        rules_file: Optional direct path to a rules YAML file.  When
            supplied this takes precedence over ``config['rules']['file']``
            so each activity can load its own rules without changing config.

    Methods
    -------
    apply_rules(attributes, obj_conf)
        Return mapping of rule name -> score for matched rules.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        rules_file: str | None = None,
    ):
        """Initialise the RuleEngine.

        Args:
            config_path: Path to the main YAML configuration file.
            rules_file: Optional override path to a rules YAML file.
                When ``None`` the path is read from
                ``config['rules']['file']`` (legacy behaviour).
        """
        self.config_path = config_path
        self.config = load_config(config_path)

        # Allow direct rules_file override; fall back to config key
        if rules_file is not None:
            self.rules_file = rules_file
        else:
            self.rules_file = self.config["rules"]["file"]

        if not os.path.exists(self.rules_file) and not os.path.isabs(self.rules_file):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            alt_path = os.path.join(config_dir, self.rules_file)
            if os.path.exists(alt_path):
                self.rules_file = alt_path

        self.rules = self._load_rules()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_rules(self) -> list:
        """Load rules from the YAML file at ``self.rules_file``.

        Returns:
            list[dict]: Parsed list of rule dictionaries.
        """
        logger.info(f"Loading rules from {self.rules_file}")
        with open(self.rules_file, "r") as f:
            rules = yaml.safe_load(f)
        logger.info(f"Loaded {len(rules)} rule(s)")
        return rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_rules(
        self,
        attributes: dict,
        obj_conf: float = 1.0,
    ) -> dict:
        """Evaluate all rules against the given attributes and return scores.

        For the ``Tall Vertical Shuttering`` rule the base score is
        multiplied by ``obj_conf`` as specified in the casting rule logic::

            if tall_vertical_shuttering > 0.5:
                score += 3 * obj_conf

        All other rules use their base score unchanged.

        Args:
            attributes: Dictionary of attribute name - value extracted from
                detection results.
            obj_conf: Mean detection confidence for the current frame or
                image.  Used only to weight the Tall Vertical Shuttering
                rule.  Defaults to ``1.0`` so existing callers that do not
                pass this argument are unaffected.

        Returns:
            dict: Mapping of rule name - score for each rule whose condition
                evaluated to True.
        """
        scores: dict[str, float] = {}

        for rule in self.rules:
            rule_name = rule.get("name", "unnamed")
            condition = rule.get("condition", {})
            base_score = rule.get("score", 0.0)

            try:
                if _evaluate_condition(condition, attributes):

                    # Confidence-weighted scoring for Tall Vertical Shuttering
                    if rule_name == "Tall Vertical Shuttering":
                        score = base_score * obj_conf
                        logger.debug(
                            f"Rule '{rule_name}' matched — "
                            f"base_score: {base_score}, "
                            f"obj_conf: {obj_conf:.4f}, "
                            f"weighted_score: {score:.4f}"
                        )
                    else:
                        score = base_score
                        logger.debug(
                            f"Rule '{rule_name}' matched — score: {score}"
                        )

                    scores[rule_name] = score

                else:
                    logger.debug(f"Rule '{rule_name}' did not match")

            except Exception as e:
                logger.error(f"Error evaluating rule '{rule_name}': {e}")

        logger.info(f"Rule scores: {scores}")
        return scores