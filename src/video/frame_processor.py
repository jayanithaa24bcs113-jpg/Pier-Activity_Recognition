"""
Frame Processor

Purpose: Frame Processor used by video pipeline.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Frame Processor used by video pipeline.

``FrameProcessor.process_frame(frame)`` saves the frame to a temporary
file, runs the image through the activity recognition pipeline, and
returns activity information and detections.

Supports multiple activities via the ``activity`` parameter which selects
the correct recogniser.  Defaults to ``reinforcement`` for full backward
compatibility.
"""

from __future__ import annotations

import cv2
import os
import tempfile

from src.activity.activity_recognizer import (
    ActivityRecognizer,
    CastingActivityRecognizer,
    CapReinforcementActivityRecognizer,
    CapCastingActivityRecognizer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Process individual video frames through the recognition pipeline.

    Args:
        activity: Activity key — ``"reinforcement"``, ``"casting"``,
            ``"cap_reinforcement"``, or ``"cap_casting"``. Defaults to
            ``"reinforcement"`` for backward compatibility.
        config_path: Path to the main YAML configuration file.
    """

    def __init__(
        self,
        activity: str = "reinforcement",
        config_path: str = "config.yaml",
    ):
        self.activity = activity
        self.config_path = config_path
        self.recognizer = self._build_recognizer()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_recognizer(self):
        """Instantiate the correct recogniser for the selected activity.

        Returns:
            ActivityRecognizer | CastingActivityRecognizer |
            CapReinforcementActivityRecognizer |
            CapCastingActivityRecognizer: Recogniser instance ready to
                process frames.
        """
        if self.activity == "casting":
            logger.info("FrameProcessor: using CastingActivityRecognizer")
            return CastingActivityRecognizer(self.config_path)

        if self.activity == "cap_reinforcement":
            logger.info(
                "FrameProcessor: using CapReinforcementActivityRecognizer"
            )
            return CapReinforcementActivityRecognizer(self.config_path)

        if self.activity == "cap_casting":
            logger.info(
                "FrameProcessor: using CapCastingActivityRecognizer"
            )
            return CapCastingActivityRecognizer(self.config_path)

        # Default — reinforcement (preserves original behaviour)
        logger.info("FrameProcessor: using ActivityRecognizer (reinforcement)")
        return ActivityRecognizer(self.config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame) -> dict:
        """Process one video frame through the recognition pipeline.

        Saves the frame to a temporary file, runs recognition, then
        removes the temporary file.

        If the underlying recogniser reports ``"Model Not Trained"``
        (placeholder-safe stage, e.g. Stage 3/4 before training), that
        label is passed through unchanged — no special handling is
        required here since it flows through the same dict shape.

        Args:
            frame: OpenCV BGR image array (numpy ndarray).

        Returns:
            dict: Keys ``activity``, ``confidence``, ``detections``,
                ``attributes``.
        """
        # ---------------------------------------------
        # Save frame temporarily
        # ---------------------------------------------
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        )
        cv2.imwrite(temp_file.name, frame)

        # ---------------------------------------------
        # Run pipeline
        # ---------------------------------------------
        output = self.recognizer.recognize(temp_file.name)

        os.remove(temp_file.name)

        confidence = 0.0
        if len(output["smoothed_scores"]) > 0:
            confidence = max(output["smoothed_scores"].values())

        return {
            "activity":   output["activity"],
            "confidence": confidence,
            "detections": output["detections"],
            "attributes": output["attributes"],
        }