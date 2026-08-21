"""
Metrics

Purpose: Training metrics extraction helpers.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Training metrics extraction helpers.

`calculate_metrics(results)` reads common YOLOv8 result fields and
returns a single-row DataFrame. Missing values are replaced with NaN
and a warning is logged.
"""

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(results) -> pd.DataFrame:
    """Calculate and return training metrics from YOLOv8 results.

    Extracts box loss, classification loss, DFL loss, precision, recall,
    mAP50, and mAP50-95 from the last training epoch.

    Args:
        results: YOLOv8 training results object returned by model.train().

    Returns:
        pd.DataFrame: Single-row DataFrame with metric columns.
    """
    try:
        # YOLOv8 stores metrics on the results object directly
        box_loss = float(results.box_loss) if hasattr(results, "box_loss") else np.nan
        cls_loss = float(results.cls_loss) if hasattr(results, "cls_loss") else np.nan
        dfl_loss = float(results.dfl_loss) if hasattr(results, "dfl_loss") else np.nan

        # Validation metrics
        precision = float(results.results_dict.get("metrics/precision(B)", np.nan))
        recall = float(results.results_dict.get("metrics/recall(B)", np.nan))
        map50 = float(results.results_dict.get("metrics/mAP50(B)", np.nan))
        map50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", np.nan))

    except Exception as e:
        logger.warning(f"Could not extract all metrics: {e}. Using NaN placeholders.")
        box_loss = cls_loss = dfl_loss = np.nan
        precision = recall = map50 = map50_95 = np.nan

    metrics = {
        "box_loss": box_loss,
        "cls_loss": cls_loss,
        "dfl_loss": dfl_loss,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
    }

    logger.info(f"Metrics: {metrics}")
    return pd.DataFrame([metrics])
