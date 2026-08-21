"""
Utils

Purpose: Detection utility helpers for pier monitoring.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Detection utility helpers for pier monitoring.

Provides bounding-box conversion, NMS, and confidence-filtering helpers
that complement the main Detector class.

Module header added to comply with coding standards. Add a concise
description of responsibilities and public API if desired.
"""

import argparse
import numpy as np
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def xyxy_to_xywh(box):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h].

    Args:
        box: Sequence of four floats (x1, y1, x2, y2).

    Returns:
        tuple: (cx, cy, w, h).
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def filter_by_confidence(results, threshold: float = 0.5):
    """Filter detection results to keep only boxes above a confidence threshold.

    Args:
        results: YOLOv8 Results list from model.predict().
        threshold: Minimum confidence score to retain. Defaults to 0.5.

    Returns:
        list[dict]: Filtered detections with keys 'box', 'conf', 'cls'.
    """
    filtered = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= threshold:
                filtered.append(
                    {
                        "box": box.xyxy[0].tolist(),
                        "conf": conf,
                        "cls": int(box.cls[0]),
                    }
                )
    return filtered


def count_objects_by_class(results, class_names: dict) -> dict:
    """Count detected objects grouped by class name.

    Args:
        results: YOLOv8 Results list from model.predict().
        class_names: Mapping of class index - class name (model.names).

    Returns:
        dict: {class_name: count}.
    """
    counts: dict[str, int] = {}
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = class_names.get(cls_id, str(cls_id))
            counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI entry-point (used for quick dataset/config validation)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detection utility helpers")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Config loaded successfully.")
    logger.info(f"  model weights : {config.get('model', {}).get('weights', 'N/A')}")
    logger.info(f"  train path    : {config.get('dataset', {}).get('train_path', 'N/A')}")
    logger.info(f"  epochs        : {config.get('training', {}).get('epochs', 'N/A')}")


if __name__ == "__main__":
    main()
