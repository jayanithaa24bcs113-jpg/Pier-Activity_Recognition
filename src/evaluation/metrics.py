"""
Metrics

Purpose: Evaluation metrics utilities.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Evaluation metrics utilities.

``MetricsCalculator`` collects true/predicted labels and computes standard
classification metrics (accuracy, precision, recall, F1, confusion matrix).

Works for any activity — operates on generic string labels with no
knowledge of which activity produced them.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class MetricsCalculator:
    """Calculate and store evaluation results for activity recognition.

    Methods
    -------
    add_result(ground_truth, prediction)
        Append one sample's ground-truth and predicted label.
    calculate_metrics()
        Compute and return a metrics dictionary.
    """

    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []

    ##########################################################

    def add_result(
        self,
        ground_truth: str,
        prediction: str,
    ) -> None:
        """Store one prediction.

        Args:
            ground_truth: True activity label string.
            prediction: Predicted activity label string.
        """
        self.true_labels.append(ground_truth)
        self.predicted_labels.append(prediction)

    ##########################################################

    def calculate_metrics(self) -> dict | None:
        """Calculate evaluation metrics.

        Returns:
            dict: Keys ``accuracy``, ``precision``, ``recall``,
                ``f1_score``, ``confusion_matrix``.
            None: When no results have been added yet.
        """
        if len(self.true_labels) == 0:
            return None

        accuracy = accuracy_score(
            self.true_labels,
            self.predicted_labels,
        )

        precision = precision_score(
            self.true_labels,
            self.predicted_labels,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            self.true_labels,
            self.predicted_labels,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            self.true_labels,
            self.predicted_labels,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(
            self.true_labels,
            self.predicted_labels,
        )

        return {
            "accuracy":         accuracy,
            "precision":        precision,
            "recall":           recall,
            "f1_score":         f1,
            "confusion_matrix": cm,
        }