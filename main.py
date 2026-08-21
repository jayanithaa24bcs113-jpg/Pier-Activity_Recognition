"""
Main

Purpose: Main CLI entry point for single-image pier monitoring inference.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Main CLI entry point for single-image pier monitoring inference.

Runs the full recognition pipeline (detect → extract → rules → bayesian
→ decide) on a single image and prints the result.

Usage
-----
    python main.py --source path/to/image.jpg
    python main.py --activity casting --source path/to/image.jpg
    python main.py --activity cap_reinforcement --source path/to/image.jpg
    python main.py --activity cap_casting --source path/to/image.jpg

When ``--activity`` is omitted it defaults to ``reinforcement`` for full
backward compatibility.

Placeholder-safe: for ``cap_reinforcement`` or ``cap_casting`` (Stages 3
and 4), if the model has not been trained yet, the recogniser returns
``"Model Not Trained"`` gracefully — this script prints that result like
any other, with no crash.
"""

import argparse
import os

from src.activity.activity_recognizer import (
    ActivityRecognizer,
    CastingActivityRecognizer,
    CapReinforcementActivityRecognizer,
    CapCastingActivityRecognizer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_recognizer(activity: str, config_path: str = "config.yaml"):
    """Return the correct recogniser for the selected activity.

    Args:
        activity: ``"reinforcement"``, ``"casting"``,
            ``"cap_reinforcement"``, or ``"cap_casting"``.
        config_path: Path to the main YAML configuration file.

    Returns:
        ActivityRecognizer | CastingActivityRecognizer |
        CapReinforcementActivityRecognizer | CapCastingActivityRecognizer
    """
    if activity == "casting":
        return CastingActivityRecognizer(config_path)
    if activity == "cap_reinforcement":
        return CapReinforcementActivityRecognizer(config_path)
    if activity == "cap_casting":
        return CapCastingActivityRecognizer(config_path)
    return ActivityRecognizer(config_path)


def main():
    """Parse CLI args, run recognition on a single image, print results."""

    parser = argparse.ArgumentParser(
        description="Run pier monitoring inference on a single image."
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="reinforcement",
        choices=["reinforcement", "casting", "cap_reinforcement", "cap_casting"],
        help="Activity to run (default: reinforcement).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        logger.error(f"Source image not found: {args.source}")
        print(f"\nError: source image not found at {args.source}\n")
        return

    recognizer = build_recognizer(args.activity, args.config)

    output = recognizer.recognize(args.source)

    print("\n" + "=" * 70)
    print(f"ACTIVITY      : {output['activity']}")
    print(f"ATTRIBUTES    : {output['attributes']}")
    print(f"RAW SCORES    : {output['raw_scores']}")
    print(f"SMOOTHED      : {output['smoothed_scores']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()