"""
Utils

Purpose: Attribute module utility helpers.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Attribute module utility helpers.

Contains helper functions for normalising and merging attribute
dictionaries used by the attribute extractor and rule engine.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalise_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp and normalise a score to [0, 1].

    Args:
        value: Raw score value.
        min_val: Expected minimum (used for linear scaling).
        max_val: Expected maximum (used for linear scaling).

    Returns:
        float: Normalised value in [0, 1].
    """
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def merge_attributes(*attribute_dicts: dict) -> dict:
    """Merge multiple attribute dictionaries by summing numeric values.

    Args:
        *attribute_dicts: Variable number of attribute dicts to merge.

    Returns:
        dict: Merged dictionary with summed values.
    """
    merged: dict = {}
    for attrs in attribute_dicts:
        for key, value in attrs.items():
            if key in merged:
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged
