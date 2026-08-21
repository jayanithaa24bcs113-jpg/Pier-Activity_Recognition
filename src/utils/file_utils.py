"""
File Utils

Purpose: File utilities used across the project.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

File utilities used across the project.

Provides small helpers for saving metrics and ensuring directories exist.
"""

import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_metrics_to_csv(metrics: pd.DataFrame, output_file: str = "outputs/metrics.csv") -> None:
    """Save a metrics DataFrame to a CSV file.

    Creates parent directories if they do not already exist.

    Args:
        metrics: DataFrame containing metric columns to save.
        output_file: Destination file path for the CSV. Defaults to 'outputs/metrics.csv'.
    """
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    metrics.to_csv(output_file, index=False)
    logger.info(f"Metrics saved to {output_file}")
    print(f"Metrics saved to {output_file}")


def ensure_dir(directory: str) -> None:
    """Create a directory (and any parents) if it does not exist.

    Args:
        directory: Path to the directory to create.
    """
    os.makedirs(directory, exist_ok=True)
