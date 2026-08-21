"""
Dataset Evaluator

Purpose: Dataset-level evaluation for pier monitoring activities.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Dataset-level evaluation for pier monitoring activities.

``DatasetEvaluator`` iterates over a labelled test dataset, runs the
full recognition pipeline on every image, and computes classification
metrics via ``ActivityEvaluator``.

Expected dataset structure
--------------------------
    datasets/<activity>/test/
        images/
            <label_class>/
                image1.jpg
                image2.jpg
                ...

The subfolder name under ``images/`` is used as the ground-truth label.
If images are not organised into subfolders a ``ground_truth_label``
fallback can be supplied at construction time.

Supports multiple activities via the ``activity`` parameter.
Defaults to ``"reinforcement"`` for backward compatibility.

Placeholder-safe: for ``"cap_reinforcement"`` or ``"cap_casting"``
(Stages 3 and 4), if the test dataset folder does not exist or is empty
(dataset not yet collected, or model not yet trained), ``evaluate()``
prints a clear message and returns ``None`` gracefully.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.activity.activity_recognizer import (
    ActivityRecognizer,
    CastingActivityRecognizer,
    CapReinforcementActivityRecognizer,
    CapCastingActivityRecognizer,
)
from src.evaluation.evaluator import ActivityEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DatasetEvaluator:
    """Run evaluation over a full test dataset.

    Args:
        activity: Activity key — ``"reinforcement"``, ``"casting"``,
            ``"cap_reinforcement"``, or ``"cap_casting"``. Defaults to
            ``"reinforcement"`` for backward compatibility.
        config_path: Path to the main YAML configuration file.
        ground_truth_label: Fallback ground-truth label used when images
            are not organised into labelled subfolders.  When ``None``
            the activity label from config.yaml is used as the fallback.
    """

    def __init__(
        self,
        activity: str = "reinforcement",
        config_path: str = "config.yaml",
        ground_truth_label: str | None = None,
    ):
        self.activity     = activity
        self.config_path  = config_path

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        activities_cfg = config.get("activities", {})
        activity_cfg   = activities_cfg.get(activity, {})

        self.activity_label = activity_cfg.get(
            "activity_label", "Pier Stem Reinforcement"
        )

        self.ground_truth_label = ground_truth_label or self.activity_label

        # Resolve test image folder
        self.test_folder = self._resolve_test_folder(config)

        # Build recogniser
        self.recognizer = self._build_recognizer()

        # Build evaluator
        self.evaluator = ActivityEvaluator(
            activity=activity,
            config_path=config_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_test_folder(self, config: dict) -> Path:
        """Resolve the test images folder for the selected activity.

        Args:
            config: Parsed config.yaml dictionary.

        Returns:
            Path: Path to the test images directory.
        """
        if self.activity == "reinforcement":
            return Path(config["dataset"]["test_path"]) / "images"

        if self.activity == "cap_reinforcement":
            # Stage 3 — read path from cap_reinforcement dataset yaml.
            # Placeholder-safe: dataset yaml exists but the dataset
            # folder itself may not exist yet.
            cap_data_yaml = config.get("cap_reinforcement_training", {}).get(
                "data", "configs/cap_reinforcement_dataset.yaml"
            )

            if os.path.exists(cap_data_yaml):
                with open(cap_data_yaml, "r", encoding="utf-8") as f:
                    cap_dataset = yaml.safe_load(f) or {}
                dataset_root = cap_dataset.get(
                    "path", "datasets/Pier_cap_reinforcement"
                )
                test_rel = cap_dataset.get("test", "images/test")
            else:
                logger.warning(
                    f"Cap reinforcement dataset yaml not found at "
                    f"{cap_data_yaml}. Falling back to default path."
                )
                dataset_root = "datasets/Pier_cap_reinforcement"
                test_rel = "images/test"

            return Path(dataset_root) / test_rel

        if self.activity == "cap_casting":
            # Stage 4 — read path from cap_casting dataset yaml.
            cap_casting_data_yaml = config.get("cap_casting_training", {}).get(
                "data", "configs/cap_casting_dataset.yaml"
            )

            if os.path.exists(cap_casting_data_yaml):
                with open(cap_casting_data_yaml, "r", encoding="utf-8") as f:
                    cap_casting_dataset = yaml.safe_load(f) or {}
                dataset_root = cap_casting_dataset.get(
                    "path", "datasets/Pier_cap_casting"
                )
                test_rel = cap_casting_dataset.get("test", "images/test")
            else:
                logger.warning(
                    f"Cap casting dataset yaml not found at "
                    f"{cap_casting_data_yaml}. Falling back to default path."
                )
                dataset_root = "datasets/Pier_cap_casting"
                test_rel = "images/test"

            return Path(dataset_root) / test_rel

        # Casting — read path from casting dataset yaml
        casting_data_yaml = config.get("casting_training", {}).get(
            "data", "configs/casting_dataset.yaml"
        )
        with open(casting_data_yaml, "r", encoding="utf-8") as f:
            casting_dataset = yaml.safe_load(f)

        dataset_root = casting_dataset.get(
            "path", "datasets/Pier_stem_casting"
        )
        test_rel = casting_dataset.get("test", "images/test")
        return Path(dataset_root) / test_rel

    def _build_recognizer(self):
        """Instantiate the correct recogniser for the selected activity.

        Returns:
            ActivityRecognizer | CastingActivityRecognizer |
            CapReinforcementActivityRecognizer |
            CapCastingActivityRecognizer
        """
        if self.activity == "casting":
            logger.info("DatasetEvaluator: using CastingActivityRecognizer")
            return CastingActivityRecognizer(self.config_path)

        if self.activity == "cap_reinforcement":
            logger.info(
                "DatasetEvaluator: using CapReinforcementActivityRecognizer"
            )
            return CapReinforcementActivityRecognizer(self.config_path)

        if self.activity == "cap_casting":
            logger.info(
                "DatasetEvaluator: using CapCastingActivityRecognizer"
            )
            return CapCastingActivityRecognizer(self.config_path)

        logger.info("DatasetEvaluator: using ActivityRecognizer (reinforcement)")
        return ActivityRecognizer(self.config_path)

    def _collect_images(self) -> list[tuple[Path, str]]:
        """Collect all test images with their ground-truth labels.

        Walks ``self.test_folder``.  When a direct child subdirectory
        exists its name is used as the ground-truth label; otherwise
        ``self.ground_truth_label`` is used as the fallback.

        Returns:
            list[tuple[Path, str]]: List of (image_path, ground_truth).
        """
        samples: list[tuple[Path, str]] = []

        if not self.test_folder.exists():
            logger.error(f"Test folder not found: {self.test_folder}")
            return samples

        # Check for labelled subfolders
        subdirs = [
            d for d in self.test_folder.iterdir() if d.is_dir()
        ]

        if subdirs:
            # Images organised into class subfolders
            for subdir in sorted(subdirs):
                label = subdir.name
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((img_path, label))
        else:
            # Flat folder — use fallback label
            for img_path in sorted(self.test_folder.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((img_path, self.ground_truth_label))

        logger.info(
            f"Collected {len(samples)} images from {self.test_folder}"
        )
        return samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> dict | None:
        """Run the full evaluation pipeline over the test dataset.

        For each image:
            1. Run the recognition pipeline
            2. Compare predicted label to ground-truth label
            3. Accumulate results in ``ActivityEvaluator``

        Then compute and save metrics.

        Placeholder-safe: if no images are found (e.g. Stage 3/4 dataset
        not yet collected, or model not yet trained so recognition
        returns "Model Not Trained" for every image), this is handled
        gracefully — an empty dataset returns None immediately, and a
        populated-but-untrained-model dataset simply evaluates against
        the "Model Not Trained" prediction for every sample rather than
        crashing.

        Returns:
            dict: Metrics dictionary, or None if no images were found.
        """
        samples = self._collect_images()

        if not samples:
            print(
                f"\nNo images found in {self.test_folder}. "
                "Check dataset path in config.yaml.\n"
            )
            return None

        print("\n")
        print("=" * 70)
        print(f"DATASET EVALUATION STARTED  —  activity: {self.activity}")
        print(f"Total images : {len(samples)}")
        print("=" * 70)

        for idx, (img_path, ground_truth) in enumerate(samples, start=1):

            print(f"\n[{idx}/{len(samples)}] {img_path.name}")

            try:
                output    = self.recognizer.recognize(str(img_path))
                predicted = output["activity"]
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                predicted = "Error"

            self.evaluator.update(ground_truth, predicted)

            match = "✓" if predicted == ground_truth else "✗"
            print(
                f"  Ground Truth : {ground_truth}"
                f"  |  Predicted : {predicted}  {match}"
            )

        print("\n")
        return self.evaluator.evaluate()