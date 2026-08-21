"""
Visualizer

Purpose: Visualizer for pier monitoring detection results.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Visualizer for pier monitoring detection results.

Draws bounding boxes and activity labels on images and either displays
them in a window or saves them to disk.

Main class `Visualizer` exposes `draw`, `display`, and `save` methods.
"""

import cv2
import os
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Colour palette (BGR) — one per class
COLOURS = {
    "rebar_cage":     (0, 200, 255),   # amber
    "vertical_rebar": (0, 128, 255),   # blue
    "stirrup":        (0, 255, 128),   # green-cyan
    "worker":         (255, 128, 0),   # orange
    "crane":          (0, 0, 220),     # red
    "pile_cap":       (200, 0, 255),   # purple
    "starter_bar":    (255, 255, 0),   # cyan
    "default":        (0, 255, 0),     # fallback green
}


class Visualizer:
    """Draw detection results and activity labels on images.

    Args:
        config_path: Path to the main YAML configuration file.
                     Used to resolve class names. Pass ``None`` to skip
                     config loading (class names fall back to indices).
    """

    def __init__(self, config_path: str = "config.yaml"):
        try:
            self.config = load_config(config_path)
            self.class_names: dict = self.config.get("classes", {})
        except FileNotFoundError:
            logger.warning(f"Config not found at '{config_path}'; class names will be numeric.")
            self.config = {}
            self.class_names = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, image_path: str, results, activity: str) -> "np.ndarray":
        """Draw boxes and activity label on an image and return it.

        Args:
            image_path: Path to the source image.
            results: YOLOv8 Results list from model.predict().
            activity: Activity string to overlay on the image.

        Returns:
            numpy.ndarray: Annotated image (BGR).
        """
        import numpy as np

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return np.zeros((640, 640, 3), dtype=np.uint8)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                colour = COLOURS.get(cls_name.lower(), COLOURS["default"])

                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
                )

        # Activity overlay
        cv2.putText(
            image, f"Activity: {activity}", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        return image

    def display(self, image_path: str, results, activity: str) -> None:
        """Draw results and show the annotated image in an OpenCV window.

        Args:
            image_path: Path to the source image.
            results: YOLOv8 Results list from model.predict().
            activity: Activity label to overlay.
        """
        image = self.draw(image_path, results, activity)
        cv2.imshow("Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, image_path: str, results, activity: str, out_path: str) -> None:
        """Draw results and save the annotated image to disk.

        Args:
            image_path: Path to the source image.
            results: YOLOv8 Results list from model.predict().
            activity: Activity label to overlay.
            out_path: Destination file path for the saved image.
        """
        image = self.draw(image_path, results, activity)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, image)
        logger.info(f"Annotated image saved to {out_path}")
