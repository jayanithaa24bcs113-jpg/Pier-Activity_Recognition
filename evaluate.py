"""
Evaluate

Purpose: CLI entry point to run dataset evaluation.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

CLI entry point to run dataset evaluation.

Constructs a ``DatasetEvaluator`` for the selected activity and triggers
evaluation over the configured test dataset.

Usage
-----
    python evaluate.py --activity casting
    python evaluate.py --activity cap_reinforcement
    python evaluate.py --activity cap_casting

When ``--activity`` is omitted it defaults to ``reinforcement`` for full
backward compatibility.

Placeholder-safe: for ``cap_reinforcement`` or ``cap_casting`` (Stages 3
and 4), if the test dataset is not yet populated, or the model is not
yet trained, ``DatasetEvaluator.evaluate()`` handles this gracefully —
no crash.
"""

import argparse

from src.evaluation.dataset_evaluator import DatasetEvaluator


def main():
    """Parse CLI args and run dataset evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate pier monitoring model on the test dataset."
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="reinforcement",
        choices=["reinforcement", "casting", "cap_reinforcement", "cap_casting"],
        help="Activity to evaluate (default: reinforcement).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )

    args = parser.parse_args()

    evaluator = DatasetEvaluator(
        activity=args.activity,
        config_path=args.config,
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()