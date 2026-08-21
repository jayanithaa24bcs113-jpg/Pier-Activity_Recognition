"""
Logger

Purpose: Logging utilities.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Logging utilities.

Provides `get_logger(name, level)` which returns a configured logger
with both console and daily-rotated file handlers (basic rotation by
date). Avoid creating duplicate handlers on repeated imports.
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "outputs", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a logger with console and file handlers.

    Args:
        name: Logger name (typically __name__).
        level: Logging level. Defaults to INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(
        LOG_DIR, f"{datetime.now().strftime('%Y%m%d')}_pier_monitoring.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
