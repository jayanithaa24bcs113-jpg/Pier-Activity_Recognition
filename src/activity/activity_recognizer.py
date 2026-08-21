"""
Activity Recognizer

Purpose: Activity Recognition Module for Pier Monitoring.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Activity Recognition Module for Pier Monitoring.

Orchestrates detection, attribute extraction, rule evaluation, and
Bayesian smoothing to produce a final activity label for an image.

Four recognisers are provided:

``ActivityRecognizer``
    Pier Stem Reinforcement (Stage 1).  Preserved exactly as before.

``CastingActivityRecognizer``
    Pier Stem Casting (Stage 2).

``CapReinforcementActivityRecognizer``
    Pier Cap Reinforcement (Stage 3).  Placeholder-safe: when
    ``activities.cap_reinforcement.model_trained`` is ``False`` in
    config.yaml, no detector/model is loaded and ``recognize()``
    returns a ``"Model Not Trained"`` result immediately.

``CapCastingActivityRecognizer``
    Pier Cap Casting (Stage 4).  Same placeholder-safe pattern as
    Stage 3, gated on ``activities.cap_casting.model_trained``.

All inherit from ``_BaseActivityRecognizer`` which implements the shared
pipeline (detect → extract → rules → bayesian → decide). Each subclass
supplies its own extractor, rule file, activity/idle labels, and decision
thresholds via ``_build_components`` and ``decide_activity``.

Each subclass also sets ``self.normalization_threshold`` — the raw
smoothed-score total at which that stage's own rules consider the
activity "confirmed". This is NOT used internally by ``decide_activity``
(each stage keeps its own bespoke decision logic), but it IS used by
``webapp/app.py`` to build a cross-stage-comparable confidence ratio
(``total_score / normalization_threshold``) when auto-detecting which
of several fired stages is the best match for an image. Without this,
stages whose rules score on a larger numeric scale (e.g. Stage 3's
1-5 point rules vs. Stage 2/4's 0.4-1.0 point rules) would win the
auto-detect comparison purely due to bigger raw numbers, not because
they are genuinely more confident.
"""

from __future__ import annotations

from src.detection.detector import Detector
from src.attributes.attribute_extractor import (
    AttributeExtractor,
    CastingAttributeExtractor,
    CapReinforcementAttributeExtractor,
    CapCastingAttributeExtractor,
)
from src.rules.rule_engine import RuleEngine
from src.bayesian.bayesian_filter import BayesianFilter
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Base class — shared pipeline, no decision logic
# ===========================================================================

class _BaseActivityRecognizer:
    """Shared pipeline for all activity recognisers.

    Subclasses must implement:
        - ``_build_components()`` — set ``self.detector``, ``self.extractor``,
          ``self.rule_engine``, ``self.activity_label``, ``self.idle_label``,
          and ``self.normalization_threshold``. May also set
          ``self.model_available = False`` when the stage's model has not
          been trained yet, in which case ``self.detector``,
          ``self.extractor``, and ``self.rule_engine`` should be left as
          ``None`` — ``recognize()`` will short-circuit before touching them.
        - ``decide_activity(attributes, scores)`` — map scores to a label

    Args:
        config_path: Path to the main YAML configuration file.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.config_path = config_path
        self.bayesian = BayesianFilter(alpha=0.4)

        # Default: model is assumed available. Stages with a placeholder
        # model (e.g. Stage 3/4 before training) override this to False
        # inside their own _build_components(). This keeps Stage 1/2
        # behaviour completely unchanged since they never set it.
        self.model_available: bool = True

        # Default normalization threshold — used by webapp/app.py to
        # build a cross-stage-comparable confidence ratio. Each subclass
        # overrides this in _build_components() with its own real
        # firing threshold. The fallback of 1.0 here is only a safety
        # net in case a future subclass forgets to set it.
        self.normalization_threshold: float = 1.0

        self._build_components()

    def _build_components(self) -> None:
        """Initialise activity-specific components.

        Must be overridden by each subclass to set:
            self.detector      : Detector | None
            self.extractor     : AttributeExtractor | CastingAttributeExtractor
                                  | CapReinforcementAttributeExtractor
                                  | CapCastingAttributeExtractor | None
            self.rule_engine   : RuleEngine | None
            self.activity_label: str
            self.idle_label    : str
            self.normalization_threshold: float
            self.model_available: bool (optional — defaults to True)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main pipeline — identical for all activities
    # ------------------------------------------------------------------

    def recognize(self, image_path: str) -> dict:
        """Run the full recognition pipeline on a single image.

        Steps:
            1. Detect objects with YOLO
            2. Extract attributes
            3. Apply rule engine
            4. Apply Bayesian smoothing
            5. Decide activity label

        If ``self.model_available`` is ``False`` (placeholder stage with
        no trained model yet), the pipeline is skipped entirely and a
        ``"Model Not Trained"`` result is returned gracefully — no
        detector, extractor, or rule engine is touched.

        Args:
            image_path: Path to the image file to analyse.

        Returns:
            dict: Keys ``activity``, ``attributes``, ``raw_scores``,
                ``smoothed_scores``, ``detections``.
        """
        if not self.model_available:
            logger.warning(
                f"Skipping recognition for '{image_path}' — "
                f"model not yet trained for this activity."
            )
            return {
                "activity": "Model Not Trained",
                "attributes": {},
                "raw_scores": {},
                "smoothed_scores": {},
                "detections": [],
            }

        logger.info(f"Processing frame: {image_path}")

        # STEP 1 : Detection
        detections = self.detector.detect(image_path)

        # STEP 2 : Attributes
        attributes = self.extractor.extract(detections)

        # STEP 3 : Rule Engine
        obj_conf = self._mean_confidence(detections)
        raw_scores = self.rule_engine.apply_rules(attributes, obj_conf)
        logger.info(f"Rule Scores : {raw_scores}")

        # STEP 4 : Bayesian
        smoothed_scores = self.bayesian.update(raw_scores)
        logger.info(f"Smoothed Scores : {smoothed_scores}")

        # STEP 5 : Activity
        activity = self.decide_activity(attributes, smoothed_scores)
        logger.info(f"Final Activity : {activity}")

        print("\n" + "=" * 70)
        print("FINAL ACTIVITY :", activity)
        print("=" * 70 + "\n")

        return {
            "activity": activity,
            "attributes": attributes,
            "raw_scores": raw_scores,
            "smoothed_scores": smoothed_scores,
            "detections": detections,
        }

    def decide_activity(self, attributes: dict, scores: dict) -> str:
        """Map smoothed scores and attributes to an activity label.

        Must be overridden by each subclass.

        Args:
            attributes: Extracted attribute dictionary.
            scores: Smoothed score dictionary from BayesianFilter.

        Returns:
            str: Human-readable activity label.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_confidence(detections) -> float:
        """Return mean detection confidence across all boxes.

        Args:
            detections: YOLOv8 Results list from ``Detector.detect()``.

        Returns:
            float: Mean confidence in [0, 1], or 1.0 if no detections.
        """
        confs = [
            float(box.conf[0])
            for result in detections
            for box in result.boxes
        ]
        return sum(confs) / len(confs) if confs else 1.0


# ===========================================================================
# Pier Stem Reinforcement Recogniser (Stage 1)
# ===========================================================================

class ActivityRecognizer(_BaseActivityRecognizer):
    """High-level pipeline coordinator for Pier Stem Reinforcement.

    Methods
    -------
    recognize(image_path)
        Run full pipeline and return activity, attributes and scores.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)

    def _build_components(self) -> None:
        """Initialise reinforcement-specific components."""
        self.detector = Detector(self.config_path, activity="reinforcement")
        self.extractor = AttributeExtractor(
            class_names=self.detector.classes
        )
        self.rule_engine = RuleEngine(self.config_path)

        # Read labels from activities block with graceful fallback
        activities_cfg = self.config.get("activities", {})
        reinforcement_cfg = activities_cfg.get("reinforcement", {})
        self.activity_label = reinforcement_cfg.get(
            "activity_label", "Pier Stem Reinforcement"
        )
        self.idle_label = reinforcement_cfg.get(
            "idle_label", "Idle / No Significant Activity"
        )

        # Cross-stage confidence normalization — this stage's decision
        # logic fires primarily via the "Pier Stem Reinforcement" rule
        # at >= 0.6, so that is the reference threshold used to build a
        # comparable confidence ratio in webapp/app.py.
        self.normalization_threshold = reinforcement_cfg.get(
            "normalization_threshold", 0.6
        )

    # ------------------------------------------------------------------
    # Decide Activity — original logic preserved exactly
    # ------------------------------------------------------------------

    def decide_activity(self, attributes: dict, scores: dict) -> str:
        """Decide reinforcement activity from smoothed scores.

        Args:
            attributes: Extracted attribute dictionary.
            scores: Smoothed score dictionary from BayesianFilter.

        Returns:
            str: Activity label.
        """
        # Rule fired directly
        if scores.get("Pier Stem Reinforcement", 0) >= 0.6:
            return self.activity_label

        # Practical construction logic
        if (
            attributes["pier_rebar_count"] >= 3
            and attributes["vertical_rebar_count"] >= 1
        ):
            return self.activity_label

        if attributes["pier_rebar_count"] >= 5:
            return self.activity_label

        if (
            attributes["vertical_rebar_count"] >= 1
            and attributes["pile_cap_count"] >= 1
        ):
            return self.activity_label

        if attributes["rebar_density"] >= 0.30:
            return self.activity_label

        return self.idle_label


# ===========================================================================
# Pier Stem Casting Recogniser (Stage 2)
# ===========================================================================

class CastingActivityRecognizer(_BaseActivityRecognizer):
    """High-level pipeline coordinator for Pier Stem Casting.

    Loads casting-specific weights, attribute extractor, and rules
    from the ``activities.casting`` block in config.yaml.

    Methods
    -------
    recognize(image_path)
        Run full pipeline and return activity, attributes and scores.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)

    def _build_components(self) -> None:
        """Initialise casting-specific components."""
        activities_cfg = self.config.get("activities", {})
        casting_cfg = activities_cfg.get("casting", {})

        self.detector = Detector(self.config_path, activity="casting")
        self.extractor = CastingAttributeExtractor(
            class_names=self.detector.classes
        )
        self.rule_engine = RuleEngine(
            self.config_path,
            rules_file=casting_cfg.get(
                "rules", "src/rules/rules_stem_casting.yaml"
            ),
        )
        self.activity_label = casting_cfg.get(
            "activity_label", "Pier Stem Casting"
        )
        self.idle_label = casting_cfg.get(
            "idle_label", "Idle / No Casting Activity"
        )
        self.score_threshold = casting_cfg.get("confidence_threshold", 0.5)

        # Cross-stage confidence normalization — this stage's general
        # threshold shortcut fires at total_score >= 1.2, so that is
        # the reference threshold used to build a comparable confidence
        # ratio in webapp/app.py.
        self.normalization_threshold = casting_cfg.get(
            "normalization_threshold", 1.2
        )

    # ------------------------------------------------------------------
    # Decide Activity — casting-specific logic
    # ------------------------------------------------------------------

    def decide_activity(self, attributes: dict, scores: dict) -> str:
        """Decide casting activity from smoothed scores and attributes.

        Casting is confirmed when:
            - ``Full Pier Stem Casting`` rule score >= 0.8  (strong composite)
            - OR ``Concrete Placement Active`` rule score >= 0.6
            - OR total smoothed score >= 1.2 AND minimum physical presence
              (formwork or concrete pump detected)

        Args:
            attributes: Extracted attribute dictionary.
            scores: Smoothed score dictionary from BayesianFilter.

        Returns:
            str: Activity label.
        """
        total_score = sum(scores.values())

        # High-confidence shortcut — strongest composite rule
        if scores.get("Full Pier Stem Casting", 0) >= 0.8:
            return self.activity_label

        # Concrete placement shortcut
        if scores.get("Concrete Placement Active", 0) >= 0.6:
            return self.activity_label

        # General threshold — enough evidence + minimum physical presence
        minimum_presence = (
            attributes.get("formwork_count", 0) >= 1
            or attributes.get("concrete_pump_count", 0) >= 1
        )

        if total_score >= 1.2 and minimum_presence:
            return self.activity_label

        return self.idle_label


# ===========================================================================
# Pier Cap Reinforcement Recogniser (Stage 3)
# ===========================================================================

class CapReinforcementActivityRecognizer(_BaseActivityRecognizer):
    """High-level pipeline coordinator for Pier Cap Reinforcement.

    Loads cap-reinforcement-specific weights, attribute extractor, and
    rules from the ``activities.cap_reinforcement`` block in config.yaml.

    Placeholder-safe: if ``activities.cap_reinforcement.model_trained``
    is ``False`` (dataset not yet collected / model not yet trained),
    ``self.model_available`` is set to ``False`` and no ``Detector`` is
    constructed — ``YOLO()`` is never asked to load a nonexistent weights
    file. Every call to ``recognize()`` then returns a ``"Model Not
    Trained"`` result immediately via the base class's guard.

    Methods
    -------
    recognize(image_path)
        Run full pipeline and return activity, attributes and scores,
        or a ``"Model Not Trained"`` result if the model isn't ready.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)

    def _build_components(self) -> None:
        """Initialise cap-reinforcement-specific components.

        Reads ``model_trained`` from config first. When ``False``, skips
        building the detector/extractor/rule engine entirely and marks
        ``self.model_available = False`` so ``recognize()`` short-circuits.
        """
        activities_cfg = self.config.get("activities", {})
        cap_cfg = activities_cfg.get("cap_reinforcement", {})

        self.activity_label = cap_cfg.get(
            "activity_label", "Pier Cap Reinforcement"
        )
        self.idle_label = cap_cfg.get(
            "idle_label", "Idle / No Cap Activity"
        )
        # Activity threshold: total smoothed rule score >= 7 → activity fires
        self.score_threshold = cap_cfg.get("score_threshold", 7)

        # Cross-stage confidence normalization — this stage's rules score
        # on a 1-5 point scale (much larger than Stage 2/4's 0.4-1.0
        # scale), so its own firing threshold (score_threshold) is
        # reused directly as the normalization reference. Without this,
        # Stage 3's raw totals would structurally dominate any
        # cross-stage "highest raw score wins" comparison regardless of
        # how confidently it actually fired relative to its own bar.
        self.normalization_threshold = self.score_threshold

        self.model_available = bool(cap_cfg.get("model_trained", False))

        if not self.model_available:
            logger.warning(
                "Stage 3 (cap_reinforcement) model_trained is False in "
                "config.yaml — skipping detector/extractor/rule_engine "
                "initialisation. Train the model first using: "
                "python train.py --activity cap_reinforcement"
            )
            self.detector = None
            self.extractor = None
            self.rule_engine = None
            return

        self.detector = Detector(self.config_path, activity="cap_reinforcement")
        self.extractor = CapReinforcementAttributeExtractor(
            class_names=self.detector.classes
        )
        self.rule_engine = RuleEngine(
            self.config_path,
            rules_file=cap_cfg.get(
                "rules", "src/rules/rules_cap_reinforcement.yaml"
            ),
        )

    def decide_activity(self, attributes: dict, scores: dict) -> str:
        """Decide cap reinforcement activity from smoothed rule scores.

        Per spec: total smoothed rule score >= 7 → ``"Pier Cap
        Reinforcement"``, else ``"Idle / No Cap Activity"``.

        Args:
            attributes: Extracted attribute dictionary (unused directly
                here; threshold is purely score-based per spec, but kept
                in the signature for interface consistency with the
                other recognisers).
            scores: Smoothed score dictionary from BayesianFilter.

        Returns:
            str: Activity label.
        """
        total_score = sum(scores.values())

        if total_score >= self.score_threshold:
            return self.activity_label

        return self.idle_label


# ===========================================================================
# Pier Cap Casting Recogniser (Stage 4)
# ===========================================================================

class CapCastingActivityRecognizer(_BaseActivityRecognizer):
    """High-level pipeline coordinator for Pier Cap Casting.

    Loads cap-casting-specific weights, attribute extractor, and rules
    from the ``activities.cap_casting`` block in config.yaml.

    Placeholder-safe: if ``activities.cap_casting.model_trained`` is
    ``False`` (model not yet trained, even though the dataset is ready),
    ``self.model_available`` is set to ``False`` and no ``Detector`` is
    constructed — ``YOLO()`` is never asked to load a nonexistent weights
    file. Every call to ``recognize()`` then returns a ``"Model Not
    Trained"`` result immediately via the base class's guard, exactly
    like ``CapReinforcementActivityRecognizer`` (Stage 3).

    Methods
    -------
    recognize(image_path)
        Run full pipeline and return activity, attributes and scores,
        or a ``"Model Not Trained"`` result if the model isn't ready.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)

    def _build_components(self) -> None:
        """Initialise cap-casting-specific components.

        Reads ``model_trained`` from config first. When ``False``, skips
        building the detector/extractor/rule engine entirely and marks
        ``self.model_available = False`` so ``recognize()`` short-circuits.
        """
        activities_cfg = self.config.get("activities", {})
        cap_casting_cfg = activities_cfg.get("cap_casting", {})

        self.activity_label = cap_casting_cfg.get(
            "activity_label", "Pier Cap Casting"
        )
        self.idle_label = cap_casting_cfg.get(
            "idle_label", "Idle / No Cap Casting Activity"
        )

        # Cross-stage confidence normalization — this stage's general
        # threshold shortcut fires at total_score >= 1.2, matching
        # Stage 2's scale, so that is the reference threshold used to
        # build a comparable confidence ratio in webapp/app.py.
        self.normalization_threshold = cap_casting_cfg.get(
            "normalization_threshold", 1.2
        )

        self.model_available = bool(cap_casting_cfg.get("model_trained", False))

        if not self.model_available:
            logger.warning(
                "Stage 4 (cap_casting) model_trained is False in "
                "config.yaml — skipping detector/extractor/rule_engine "
                "initialisation. Train the model first using: "
                "python train.py --activity cap_casting"
            )
            self.detector = None
            self.extractor = None
            self.rule_engine = None
            return

        self.detector = Detector(self.config_path, activity="cap_casting")
        self.extractor = CapCastingAttributeExtractor(
            class_names=self.detector.classes
        )
        self.rule_engine = RuleEngine(
            self.config_path,
            rules_file=cap_casting_cfg.get(
                "rules", "src/rules/rules_cap_casting.yaml"
            ),
        )

    # ------------------------------------------------------------------
    # Decide Activity — cap-casting-specific logic (mirrors Stage 2's
    # shortcut structure, per Step 0 agreement)
    # ------------------------------------------------------------------

    def decide_activity(self, attributes: dict, scores: dict) -> str:
        """Decide cap casting activity from smoothed scores and attributes.

        Cap casting is confirmed when:
            - ``Full Pier Cap Casting`` rule score >= 0.8  (strong composite)
            - OR ``Concrete Placement Active`` rule score >= 0.6
            - OR total smoothed score >= 1.2 AND minimum physical presence
              (cap formwork or concrete pump detected)

        Args:
            attributes: Extracted attribute dictionary.
            scores: Smoothed score dictionary from BayesianFilter.

        Returns:
            str: Activity label.
        """
        total_score = sum(scores.values())

        # High-confidence shortcut — strongest composite rule
        if scores.get("Full Pier Cap Casting", 0) >= 0.8:
            return self.activity_label

        # Concrete placement shortcut
        if scores.get("Concrete Placement Active", 0) >= 0.6:
            return self.activity_label

        # General threshold — enough evidence + minimum physical presence
        minimum_presence = (
            attributes.get("cap_formwork_count", 0) >= 1
            or attributes.get("concrete_pump_count", 0) >= 1
        )

        if total_score >= 1.2 and minimum_presence:
            return self.activity_label

        return self.idle_label