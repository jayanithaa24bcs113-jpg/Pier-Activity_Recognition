"""
Evaluator

Purpose: Activity evaluation utilities.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Activity evaluation utilities.

Provides ``ActivityEvaluator`` which wraps ``MetricsCalculator`` to
accumulate predictions and generate summary metrics.

Supports multiple activities via the ``activity`` parameter which routes
output to the correct directory defined in config.yaml.  Defaults to
``"reinforcement"`` for full backward compatibility.
"""

from __future__ import annotations

import csv
import os

import yaml

from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActivityEvaluator:
    """Activity Evaluation Engine.

    Use ``update(ground_truth, prediction)`` to add a single prediction.
    Call ``evaluate()`` to compute and persist metrics.

    Args:
        activity: Activity key — ``"reinforcement"`` or ``"casting"``.
            Defaults to ``"reinforcement"`` for backward compatibility.
        config_path: Path to the main YAML configuration file.
    """

    def __init__(
        self,
        activity: str = "reinforcement",
        config_path: str = "config.yaml",
    ):
        self.activity = activity
        self.metrics = MetricsCalculator()

        # Resolve output folder from config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        activities_cfg = config.get("activities", {})
        activity_cfg   = activities_cfg.get(activity, {})

        self.output_folder = activity_cfg.get(
            "output_dir", "outputs"
        )

        os.makedirs(self.output_folder, exist_ok=True)

    ############################################################

    def update(
        self,
        ground_truth: str,
        prediction: str,
    ) -> None:
        """Store one prediction.

        Args:
            ground_truth: True activity label string.
            prediction: Predicted activity label string.
        """
        self.metrics.add_result(ground_truth, prediction)

    ############################################################

    def evaluate(self) -> dict | None:
        """Calculate and display evaluation metrics.

        Returns:
            dict: Metrics dictionary from ``MetricsCalculator``.
            None: When no evaluation data is available.
        """
        results = self.metrics.calculate_metrics()

        if results is None:
            print("\nNo evaluation data available.\n")
            return None

        print("\n")
        print("=" * 60)
        print(f"   ACTIVITY RECOGNITION EVALUATION — {self.activity.upper()}")
        print("=" * 60)
        print(f"Accuracy  : {results['accuracy']  * 100:.2f}%")
        print(f"Precision : {results['precision'] * 100:.2f}%")
        print(f"Recall    : {results['recall']    * 100:.2f}%")
        print(f"F1 Score  : {results['f1_score']  * 100:.2f}%")
        print("\nConfusion Matrix\n")
        print(results["confusion_matrix"])
        print("=" * 60)

        self.save_csv(results)

        return results

    ############################################################

    def save_csv(self, results: dict) -> None:
        """Save evaluation metrics to CSV in the activity output folder.

        Args:
            results: Metrics dictionary from ``MetricsCalculator``.
        """
        csv_path = os.path.join(
            self.output_folder,
            "evaluation_metrics.csv",
        )

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Activity", "Metric", "Value"])
            writer.writerow([self.activity, "Accuracy",  results["accuracy"]])
            writer.writerow([self.activity, "Precision", results["precision"]])
            writer.writerow([self.activity, "Recall",    results["recall"]])
            writer.writerow([self.activity, "F1 Score",  results["f1_score"]])

        logger.info(f"Evaluation metrics saved to {csv_path}")