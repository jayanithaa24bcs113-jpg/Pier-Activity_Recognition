"""
Config Loader

Purpose: Configuration loader utility.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Configuration loader utility.

Provide `load_config(config_path)` that reads YAML and returns a
configuration dictionary. Raises `FileNotFoundError` if the file is
missing to fail-fast during startup.
"""

import yaml
import os


def load_config(config_path="config.yaml"):
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
