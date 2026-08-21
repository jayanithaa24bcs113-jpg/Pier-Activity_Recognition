"""
Detector

Purpose: Object detection module for Pier Monitoring.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Object detection module for Pier Monitoring.

Wraps a YOLOv8 model loaded from config.yaml.  Supports multi-activity
mode via the ``activity`` parameter, which selects the correct weights
and class names from the ``activities`` block in config.yaml.  When no
activity is supplied the legacy ``model.weights`` path is used so all
existing code continues to work without modification.

Placeholder-safe: if the resolved weights file does not exist on disk
(e.g. Stage 3 before training completes), ``_load_model`` logs a clear
warning and returns ``None`` instead of raising. ``detect()`` then
returns an empty detections list rather than crashing.
"""

import os

from ultralytics import YOLO
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Detector:
    """YOLOv8-based object detector.

    Args:
        config_path: Path to the main YAML configuration file.
        activity: Activity key defined under ``activities`` in config.yaml
            (e.g. ``"reinforcement"`` or ``"casting"``).  When ``None``
            the legacy ``model.weights`` entry is used.

    Attributes:
        classes: Dict mapping class index to class name string.
        model: Loaded YOLO instance, or ``None`` if the weights file
            does not exist on disk (placeholder-safe mode).
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        activity: str | None = None,
    ):
        self.config = load_config(config_path)
        self.activity = activity
        self.classes = self._load_classes()
        self.model = self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Resolve the weights path for the current activity from config.

        Returns:
            str: Path to the weights file (may or may not exist on disk).
        """
        if (
            self.activity is not None
            and "activities" in self.config
            and self.activity in self.config["activities"]
        ):
            return self.config["activities"][self.activity]["weights"]

        # Legacy fallback — keeps existing behaviour unchanged
        return self.config["model"]["weights"]

    def _load_model(self):
        """Load YOLOv8 model weights from config.

        Checks whether the resolved weights file exists on disk before
        attempting to load it. This makes the detector placeholder-safe
        for activities whose model has not been trained yet (e.g. Stage 3
        prior to dataset collection and training).

        Returns:
            YOLO: Loaded model instance, if the weights file exists.
            None: If the weights file does not exist on disk. A warning
                is logged; no exception is raised.
        """
        model_path = self._resolve_model_path()

        if not os.path.exists(model_path) and not os.path.isabs(model_path):
            config_dir = os.path.dirname(os.path.abspath(self.config_path if hasattr(self, 'config_path') else "config.yaml"))
            alt_path = os.path.join(config_dir, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path

        if not os.path.exists(model_path):
            logger.warning(
                "Stage 3 weights not found at %s. "
                "Train the model first using: python train.py --activity cap_reinforcement",
                model_path,
            ) if self.activity == "cap_reinforcement" else logger.warning(
                "Weights not found at %s for activity '%s'. "
                "Detector will return empty detections until the model is trained.",
                model_path,
                self.activity,
            )
            return None

        logger.info(f"Loading model from {model_path}")
        return YOLO(model_path)

    def _load_classes(self) -> dict:
        """Load class index - name mapping from config.

        When an activity is specified the class map is read from the
        ``activities`` block so it always matches the trained model.
        Otherwise the top-level ``classes`` block is used (legacy path).

        This is resolved independently of whether the model weights
        actually exist on disk, so ``self.classes`` is always available
        even in placeholder-safe mode (e.g. for UI display or logging
        before a model has been trained).

        Returns:
            dict: Mapping of ``{int: str}`` class index to name.
        """
        if (
            self.activity is not None
            and "activities" in self.config
            and self.activity in self.config["activities"]
        ):
            raw = self.config["activities"][self.activity]["classes"]
        else:
            raw = self.config.get("classes", {})

        return {int(k): v for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: str):
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            list: YOLOv8 Results objects (one per image passed to predict),
                or an empty list if no model is loaded (placeholder-safe
                mode — weights file not found on disk).
        """
        if self.model is None:
            logger.warning(
                f"No model loaded for activity '{self.activity}' — "
                f"skipping inference on {image_path} and returning "
                f"empty detections."
            )
            return []

        results = self.model.predict(
            image_path,
            device="cpu",
            verbose=True
        )

        logger.debug("========== MODEL CLASSES ==========")
        logger.debug(self.classes)

        logger.debug("========== DETECTIONS ==========")

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                logger.debug(f"{cls}  {self.classes.get(cls, 'unknown')}  {round(conf, 3)}")

        return results

    def verify_dataset(self, dataset_path: str) -> bool:
        """Verify that a dataset directory has the expected structure.

        Args:
            dataset_path: Root path of the dataset to verify.

        Returns:
            bool: ``True`` if the structure is valid, ``False`` otherwise.
        """
        import os

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path {dataset_path} does not exist")
            return False

        if not os.path.exists(os.path.join(dataset_path, "images")):
            logger.error("Missing images directory")
            return False

        if not os.path.exists(os.path.join(dataset_path, "labels")):
            logger.error("Missing labels directory")
            return False

        logger.info("Dataset structure verified")
        return True