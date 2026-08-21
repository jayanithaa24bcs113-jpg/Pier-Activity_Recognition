"""
Bayesian Filter

Purpose: Bayesian exponential smoothing filter for activity scores.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Bayesian exponential smoothing filter for activity scores.

Maintains a running smoothed estimate of each named score using an
exponential moving average (EMA), which acts as a Bayesian update
with a uniform prior.

Module header added to satisfy coding standards. Class docstring
describes purpose; methods contain inline docstrings where needed.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

class BayesianFilter:
    """Exponential-smoothing filter for per-activity confidence scores.

    Each call to :meth:`update` blends the new raw scores with the existing
    smoothed estimates::

        smoothed[k] = alpha * new[k] + (1 - alpha) * smoothed[k]

    Args:
        alpha: Smoothing factor in (0, 1].  Higher values give more weight
               to recent observations; 1.0 disables smoothing entirely.
    """

    def __init__(self, alpha: float = 0.5):
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.smoothed_scores: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, scores: dict) -> dict:
        """Incorporate new raw scores and return the updated smoothed scores.

        For keys not yet seen, the raw value is accepted as-is (first
        observation).

        Args:
            scores: Dictionary of {activity_name: raw_score} from the rule engine.

        Returns:
            dict: Updated smoothed scores for all tracked activities.
        """
        for key, value in scores.items():
            if key in self.smoothed_scores:
                prev = self.smoothed_scores[key]
                self.smoothed_scores[key] = self.alpha * value + (1 - self.alpha) * prev
                logger.debug(
                    f"[{key}] smoothed: {prev:.4f} - {self.smoothed_scores[key]:.4f}"
                    f" (raw={value:.4f}, α={self.alpha})"
                )
            else:
                self.smoothed_scores[key] = value
                logger.debug(f"[{key}] initialised at {value:.4f}")

        logger.info(f"Smoothed scores: {self.smoothed_scores}")
        return self.smoothed_scores

    def reset(self) -> None:
        """Clear all smoothed scores (start fresh)."""
        self.smoothed_scores = {}
        logger.info("BayesianFilter reset.")