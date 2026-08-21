"""
Attribute Extractor

Purpose: Attribute extractor for Pier Monitoring activities.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Attribute extractor for Pier Monitoring activities.

Extracts high-level attributes from YOLO detections for activity recognition.

Four extractors are provided:

``AttributeExtractor``
    Pier Stem Reinforcement (Stage 1).
    Dataset classes: Crane, Pile cap, pier rebar, vertical rebars, worker.

``CastingAttributeExtractor``
    Pier Stem Casting (Stage 2).
    Dataset classes: Formwork, Concrete Pump, Transit Mixer, Vibrator,
    Fresh Concrete, Worker, Casted Pier.

``CapReinforcementAttributeExtractor``
    Pier Cap Reinforcement (Stage 3).
    Dataset classes: horizontal_rebar, pier_stem_top, rebar_cage, worker, crane.

``CapCastingAttributeExtractor``
    Pier Cap Casting (Stage 4).
    Dataset classes: Casted cap, Concrete pump, cap formwork,
    needle vibrator, pier stem, transit mixer, worker.

All expose an ``extract(results)`` method that returns a dictionary of
named attributes consumed by the rule engine.
"""

from __future__ import annotations

import math
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# Dataset Class Names — Reinforcement (Stage 1)
# ------------------------------------------------------------

CLASS_CRANE = "crane"
CLASS_PILE_CAP = "pile cap"
CLASS_PIER_REBAR = "pier rebar"
CLASS_VERTICAL_REBAR = "vertical rebars"
CLASS_WORKER = "worker"

REBAR_CLASSES = {
    CLASS_PIER_REBAR,
    CLASS_VERTICAL_REBAR,
}

# ------------------------------------------------------------
# Dataset Class Names — Casting (Stage 2)
# ------------------------------------------------------------

CLASS_FORMWORK = "formwork"
CLASS_CONCRETE_PUMP = "concrete pump"
CLASS_TRANSIT_MIXER = "transit mixer"
CLASS_VIBRATOR = "vibrator"
CLASS_FRESH_CONCRETE = "fresh concrete"
CLASS_CASTING_WORKER = "worker"
CLASS_CASTED_PIER = "casted pier"

CASTING_CLASSES = {
    CLASS_FORMWORK,
    CLASS_CONCRETE_PUMP,
    CLASS_TRANSIT_MIXER,
    CLASS_VIBRATOR,
    CLASS_FRESH_CONCRETE,
    CLASS_CASTING_WORKER,
    CLASS_CASTED_PIER,
}

# ------------------------------------------------------------
# Dataset Class Names — Pier Cap Reinforcement (Stage 3)
# ------------------------------------------------------------

CLASS_HORIZONTAL_REBAR = "horizontal rebar"
CLASS_PIER_STEM_TOP = "pier stem"
CLASS_REBAR_CAGE = "rebar_cage"
CLASS_CAP_WORKER = "worker"
CLASS_CAP_CRANE = "crane"

CAP_REINFORCEMENT_CLASSES = {
    CLASS_HORIZONTAL_REBAR,
    CLASS_PIER_STEM_TOP,
    CLASS_REBAR_CAGE,
    CLASS_CAP_WORKER,
    CLASS_CAP_CRANE,
}

# ------------------------------------------------------------
# Dataset Class Names — Pier Cap Casting (Stage 4)
# ------------------------------------------------------------

CLASS_CASTED_CAP = "casted cap"
CLASS_CAP_CASTING_CONCRETE_PUMP = "concrete pump"
CLASS_CAP_FORMWORK = "cap formwork"
CLASS_CAP_CASTING_VIBRATOR = "needle vibrator"
CLASS_PIER_STEM = "pier stem"
CLASS_CAP_CASTING_TRANSIT_MIXER = "transit mixer"
CLASS_CAP_CASTING_WORKER = "worker"

CAP_CASTING_CLASSES = {
    CLASS_CASTED_CAP,
    CLASS_CAP_CASTING_CONCRETE_PUMP,
    CLASS_CAP_FORMWORK,
    CLASS_CAP_CASTING_VIBRATOR,
    CLASS_PIER_STEM,
    CLASS_CAP_CASTING_TRANSIT_MIXER,
    CLASS_CAP_CASTING_WORKER,
}

# ------------------------------------------------------------
# Shared Helper Functions
# ------------------------------------------------------------

def box_area(box) -> float:
    """Return the pixel area of a bounding box.

    Args:
        box: Sequence of (x1, y1, x2, y2) coordinates.

    Returns:
        float: Area in pixels, clamped to zero for degenerate boxes.
    """
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def centroid_distance(box1, box2) -> float:
    """Return the Euclidean distance between the centroids of two boxes.

    Args:
        box1: Sequence of (x1, y1, x2, y2) for the first box.
        box2: Sequence of (x1, y1, x2, y2) for the second box.

    Returns:
        float: Pixel distance between centroids.
    """
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2

    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def any_worker_near(
    worker_boxes: list,
    target_boxes: list,
    proximity_threshold: float = 200.0,
) -> bool:
    """Return True if any worker centroid is within threshold of any target box.

    Shared by both extractors so proximity logic is defined exactly once.

    Args:
        worker_boxes: List of (x1, y1, x2, y2) worker bounding boxes.
        target_boxes: List of (x1, y1, x2, y2) target object bounding boxes.
        proximity_threshold: Maximum centroid distance in pixels to count
            as "near".  Defaults to 200.

    Returns:
        bool: True if at least one worker is near at least one target object.
    """
    for worker in worker_boxes:
        for target in target_boxes:
            if centroid_distance(worker, target) < proximity_threshold:
                return True
    return False


def mean_aspect_ratio(boxes: list) -> float:
    """Return the mean width/height aspect ratio across a list of boxes.

    Shared helper — used by Stage 3 for both ``avg_rebar_aspect_ratio``
    and ``avg_cage_aspect_ratio`` so the computation is defined once.

    Args:
        boxes: List of (x1, y1, x2, y2) bounding boxes.

    Returns:
        float: Mean of (width / height) across all boxes. Returns 0.0
            when the list is empty, guarding against division by zero.
    """
    if not boxes:
        return 0.0

    ratios = []
    for x1, y1, x2, y2 in boxes:
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        ratios.append(w / h)

    return sum(ratios) / len(ratios)


def evaluate_cap_stem_relationship(cap_boxes: list, stem_boxes: list) -> dict:
    """Evaluate structural relationship between cap and stem."""
    if not cap_boxes or not stem_boxes:
        return {
            "cap_overhang_ratio": 0.0,
            "cap_wider_than_stem": False,
            "cap_above_stem": False,
            "cap_horizontally_aligned_with_stem": False,
            "cap_stem_relationship_confirmed": False,
        }
    best_pair=None
    best_overlap=-1.0
    for cap_box in cap_boxes:
        for stem_box in stem_boxes:
            ox1=max(cap_box[0],stem_box[0]); ox2=min(cap_box[2],stem_box[2])
            overlap=max(0.0,ox2-ox1)
            if overlap>best_overlap:
                best_overlap=overlap; best_pair=(cap_box,stem_box)
    cap_box,stem_box=best_pair
    cw=max(cap_box[2]-cap_box[0],1e-6); sw=max(stem_box[2]-stem_box[0],1e-6)
    ratio=cw/sw
    cabove=((cap_box[1]+cap_box[3])/2)<((stem_box[1]+stem_box[3])/2)
    aligned=best_overlap>0
    wider=ratio>1.0
    return {
        "cap_overhang_ratio":round(ratio,4),
        "cap_wider_than_stem":wider,
        "cap_above_stem":cabove,
        "cap_horizontally_aligned_with_stem":aligned,
        "cap_stem_relationship_confirmed":wider and cabove and aligned,
    }


# ------------------------------------------------------------
# Attribute Extractor — Reinforcement (Stage 1)
# ------------------------------------------------------------

class AttributeExtractor:
    """Extract numerical attributes from YOLO detection results.

    Parameters
    ----------
    class_names: dict
        Mapping from class id to class name (model.names).
    image_width, image_height: int
        Expected image dimensions for normalising area-based attributes.
    """

    def __init__(self, class_names, image_width=640, image_height=640):

        self.class_names = {
            int(k): v.lower()
            for k, v in class_names.items()
        }

        self.image_area = image_width * image_height

    def extract(self, results):

        counts = {
            CLASS_CRANE: 0,
            CLASS_PILE_CAP: 0,
            CLASS_PIER_REBAR: 0,
            CLASS_VERTICAL_REBAR: 0,
            CLASS_WORKER: 0,
        }

        rebar_boxes = []
        worker_boxes = []

        rebar_area = 0

        # ----------------------------------------------------
        # Read all detections
        # ----------------------------------------------------

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                cls_name = self.class_names.get(cls, "").lower()

                xyxy = box.xyxy[0].tolist()

                if cls_name in counts:
                    counts[cls_name] += 1

                if cls_name in REBAR_CLASSES:
                    rebar_boxes.append(xyxy)
                    rebar_area += box_area(xyxy)

                if cls_name == CLASS_WORKER:
                    worker_boxes.append(xyxy)

        # ----------------------------------------------------
        # Rebar Density
        # ----------------------------------------------------

        rebar_density = min(rebar_area / self.image_area, 1.0)

        # ----------------------------------------------------
        # Worker Near Rebar
        # ----------------------------------------------------

        worker_near_rebar = int(
            any_worker_near(worker_boxes, rebar_boxes)
        )

        # ----------------------------------------------------
        # Final Attributes
        # ----------------------------------------------------

        attributes = {

            "crane_count":
                counts[CLASS_CRANE],

            "pile_cap_count":
                counts[CLASS_PILE_CAP],

            "pier_rebar_count":
                counts[CLASS_PIER_REBAR],

            "vertical_rebar_count":
                counts[CLASS_VERTICAL_REBAR],

            "worker_count":
                counts[CLASS_WORKER],

            "rebar_density":
                rebar_density,

            "worker_near_rebar":
                worker_near_rebar,

            "total_objects":
                sum(counts.values()),
        }

        logger.info("========== ATTRIBUTES ==========")
        logger.info(attributes)

        return attributes


# ------------------------------------------------------------
# Attribute Extractor — Casting (Stage 2)
# ------------------------------------------------------------

class CastingAttributeExtractor:
    """Extract numerical attributes from YOLO detections for Pier Stem Casting.

    Mirrors the interface of ``AttributeExtractor`` exactly: construct with
    ``class_names`` and optional image dimensions, then call
    ``extract(results)`` to receive a flat attribute dictionary.

    Parameters
    ----------
    class_names: dict
        Mapping from class id to class name (model.names).
    image_width, image_height: int
        Image dimensions used to normalise area-based attributes.
    """

    def __init__(
        self,
        class_names: dict,
        image_width: int = 640,
        image_height: int = 640,
    ):
        self.class_names = {
            int(k): v.lower()
            for k, v in class_names.items()
        }
        self.image_area = image_width * image_height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, results) -> dict:
        """Extract casting-specific attributes from YOLO results.

        Args:
            results: List of YOLOv8 Results objects returned by
                ``model.predict()``.

        Returns:
            dict: Named attributes for the rule engine, including counts,
                proximity flags, density, and aspect-ratio metrics.
        """

        counts = {
            CLASS_FORMWORK :0,
            CLASS_CONCRETE_PUMP : 0,
            CLASS_TRANSIT_MIXER : 0,
            CLASS_VIBRATOR : 0,
            CLASS_FRESH_CONCRETE : 0,
            CLASS_CASTING_WORKER : 0,
            CLASS_CASTED_PIER : 0
        }

        formwork_boxes: list = []
        fresh_concrete_boxes: list = []
        worker_boxes: list = []
        fresh_concrete_area: float = 0.0

        # aspect ratios of formwork boxes — used for tall shuttering
        formwork_aspect_ratios: list[float] = []

        # ------------------------------------------------------------------
        # Read all detections
        # ------------------------------------------------------------------

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                cls_name = self.class_names.get(cls, "").lower()
                xyxy = box.xyxy[0].tolist()

                if cls_name in counts:
                    counts[cls_name] += 1

                if cls_name == CLASS_FORMWORK:
                    formwork_boxes.append(xyxy)
                    x1, y1, x2, y2 = xyxy
                    w = max(x2 - x1, 1)
                    h = max(y2 - y1, 1)
                    formwork_aspect_ratios.append(h / w)

                if cls_name == CLASS_FRESH_CONCRETE:
                    fresh_concrete_boxes.append(xyxy)
                    fresh_concrete_area += box_area(xyxy)

                if cls_name == CLASS_CASTING_WORKER:
                    worker_boxes.append(xyxy)

        # ------------------------------------------------------------------
        # Concrete Density
        # fresh_concrete pixel area as a fraction of the image area
        # ------------------------------------------------------------------

        concrete_density = min(
            fresh_concrete_area / self.image_area, 1.0
        )

        # ------------------------------------------------------------------
        # Proximity flags — reuse shared helper
        # ------------------------------------------------------------------

        worker_near_formwork: bool = any_worker_near(
            worker_boxes, formwork_boxes
        )

        worker_near_concrete: bool = any_worker_near(
            worker_boxes, fresh_concrete_boxes
        )

        # ------------------------------------------------------------------
        # Tall Vertical Shuttering
        # Maximum height/width ratio across all formwork boxes.
        # A ratio > 0.5 indicates tall vertical formwork (used in Rule 9).
        # ------------------------------------------------------------------

        tall_vertical_shuttering: float = (
            max(formwork_aspect_ratios) if formwork_aspect_ratios else 0.0
        )

        # ------------------------------------------------------------------
        # Final Attributes
        # ------------------------------------------------------------------

        attributes = {

            "formwork_count":
                counts[CLASS_FORMWORK],

            "concrete_pump_count":
                counts[CLASS_CONCRETE_PUMP],

            "transit_mixer_count":
                counts[CLASS_TRANSIT_MIXER],

            "vibrator_count":
                counts[CLASS_VIBRATOR],

            "fresh_concrete_count":
                counts[CLASS_FRESH_CONCRETE],

            "casted_pier_count":
                counts[CLASS_CASTED_PIER],

            "worker_count":
                counts[CLASS_CASTING_WORKER],

            "concrete_density":
                round(concrete_density, 4),

            "worker_near_formwork":
                worker_near_formwork,

            "worker_near_concrete":
                worker_near_concrete,

            "tall_vertical_shuttering":
                round(tall_vertical_shuttering, 4),

            "total_objects":
                sum(counts.values()),
        }

        logger.info("========== CASTING ATTRIBUTES ==========")
        logger.info(attributes)

        return attributes


# ------------------------------------------------------------
# Attribute Extractor — Pier Cap Reinforcement (Stage 3)
# ------------------------------------------------------------

class CapReinforcementAttributeExtractor:
    """Extract numerical attributes from YOLO detections for Pier Cap Reinforcement.

    Mirrors the interface of ``AttributeExtractor`` and
    ``CastingAttributeExtractor`` exactly: construct with ``class_names``
    and optional image dimensions, then call ``extract(results)`` to
    receive a flat attribute dictionary.

    Geometric context (used by the rule engine):
        - ``horizontal_rebar``: flat rebar mat, W:H ratio > 4, expected to
          sit on top of the pier stem.
        - ``pier_stem_top``: top surface of the completed concrete pier
          stem; confirms the stage prerequisite.
        - ``rebar_cage``: shallow/wide cage geometry, H:W ratio < 0.5
          (equivalently W:H ratio > 2), distinct from Stage 1's tall
          narrow rebar cage.

    Parameters
    ----------
    class_names: dict
        Mapping from class id to class name (model.names or config
        activities.cap_reinforcement.classes).
    image_width, image_height: int
        Image dimensions used to normalise area-based attributes.
    """

    def __init__(
        self,
        class_names: dict,
        image_width: int = 640,
        image_height: int = 640,
    ):
        self.class_names = {
            int(k): v.lower()
            for k, v in class_names.items()
        }
        self.image_area = image_width * image_height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, results) -> dict:
        """Extract cap-reinforcement-specific attributes from YOLO results.

        Args:
            results: List of YOLOv8 Results objects returned by
                ``model.predict()``.

        Returns:
            dict: Named attributes for the rule engine, including counts,
                density, aspect-ratio metrics, and proximity flag.
        """

        counts = {
            CLASS_HORIZONTAL_REBAR: 0,
            CLASS_PIER_STEM_TOP: 0,
            CLASS_REBAR_CAGE: 0,
            CLASS_CAP_WORKER: 0,
            CLASS_CAP_CRANE: 0,
        }

        horizontal_rebar_boxes: list = []
        rebar_cage_boxes: list = []
        pier_stem_top_boxes: list = []
        worker_boxes: list = []

        # ------------------------------------------------------------------
        # Read all detections
        # ------------------------------------------------------------------

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                cls_name = self.class_names.get(cls, "").lower()
                xyxy = box.xyxy[0].tolist()

                if cls_name in counts:
                    counts[cls_name] += 1

                if cls_name == CLASS_HORIZONTAL_REBAR:
                    horizontal_rebar_boxes.append(xyxy)

                if cls_name == CLASS_REBAR_CAGE:
                    rebar_cage_boxes.append(xyxy)

                if cls_name == CLASS_PIER_STEM_TOP:
                    pier_stem_top_boxes.append(xyxy)

                if cls_name == CLASS_CAP_WORKER:
                    worker_boxes.append(xyxy)

        # ------------------------------------------------------------------
        # Rebar Density
        # (horizontal_rebar_count + rebar_cage_count) / total_objects
        # Guarded against division by zero.
        # ------------------------------------------------------------------

        total_objects = sum(counts.values())

        rebar_density = (
            (counts[CLASS_HORIZONTAL_REBAR] + counts[CLASS_REBAR_CAGE])
            / total_objects
            if total_objects > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Aspect Ratios — shared helper, computed once per geometry class
        # ------------------------------------------------------------------

        avg_rebar_aspect_ratio = mean_aspect_ratio(horizontal_rebar_boxes)
        avg_cage_aspect_ratio = mean_aspect_ratio(rebar_cage_boxes)

        # ------------------------------------------------------------------
        # Worker Near Rebar — reuse shared proximity helper exactly as
        # used by Stage 1 and Stage 2. Considers proximity to either
        # horizontal_rebar or rebar_cage detections.
        # ------------------------------------------------------------------

        worker_near_rebar: bool = any_worker_near(
            worker_boxes, horizontal_rebar_boxes + rebar_cage_boxes
        )

        # ------------------------------------------------------------------
        # Cap-Stem Structural Relationship — geometric disambiguation
        # between genuine pier cap activity and pier stem activity.
        # Evaluated against both horizontal_rebar and rebar_cage boxes
        # combined, since either can represent the cap-side object.
        # ------------------------------------------------------------------

        cap_stem_relationship = evaluate_cap_stem_relationship(
            horizontal_rebar_boxes + rebar_cage_boxes,
            pier_stem_top_boxes,
        )

        # ------------------------------------------------------------------
        # Final Attributes
        # ------------------------------------------------------------------

        attributes = {

            "horizontal_rebar_count":
                counts[CLASS_HORIZONTAL_REBAR],

            "pier_stem_top_count":
                counts[CLASS_PIER_STEM_TOP],

            "rebar_cage_count":
                counts[CLASS_REBAR_CAGE],

            "worker_count":
                counts[CLASS_CAP_WORKER],

            "crane_count":
                counts[CLASS_CAP_CRANE],

            "rebar_density":
                round(rebar_density, 4),

            "avg_rebar_aspect_ratio":
                round(avg_rebar_aspect_ratio, 4),

            "avg_cage_aspect_ratio":
                round(avg_cage_aspect_ratio, 4),

            "worker_near_rebar":
                worker_near_rebar,

            "cap_overhang_ratio": cap_stem_relationship["cap_overhang_ratio"],
            "cap_wider_than_stem": cap_stem_relationship["cap_wider_than_stem"],
            "cap_above_stem": cap_stem_relationship["cap_above_stem"],
            "cap_horizontally_aligned_with_stem": cap_stem_relationship["cap_horizontally_aligned_with_stem"],
            "cap_stem_relationship_confirmed": cap_stem_relationship["cap_stem_relationship_confirmed"],
            "total_objects":
                total_objects,
        }

        logger.info("========== CAP REINFORCEMENT ATTRIBUTES ==========")
        logger.info(attributes)

        return attributes


# ------------------------------------------------------------
# Attribute Extractor — Pier Cap Casting (Stage 4)
# ------------------------------------------------------------

class CapCastingAttributeExtractor:
    """Extract numerical attributes from YOLO detections for Pier Cap Casting.

    Mirrors the interface of ``CastingAttributeExtractor`` (Stage 2)
    exactly, adapted for the Stage 4 class set. Unlike Stage 2, this
    dataset has no "fresh concrete" class — ``casted_cap`` (the
    finished/curing concrete surface) is used as the density proxy
    instead, since it is the closest available visual signal for
    concrete presence in this dataset.

    Geometric context (used by the rule engine):
        - ``cap formwork``: shuttering/mould assembled around the cap
          before pouring.
        - ``pier stem``: confirms the stage prerequisite — the stem the
          cap sits on top of must be visible.
        - ``casted cap``: the poured/curing concrete cap surface.

    Parameters
    ----------
    class_names: dict
        Mapping from class id to class name (model.names or config
        activities.cap_casting.classes).
    image_width, image_height: int
        Image dimensions used to normalise area-based attributes.
    """

    def __init__(
        self,
        class_names: dict,
        image_width: int = 640,
        image_height: int = 640,
    ):
        self.class_names = {
            int(k): v.lower()
            for k, v in class_names.items()
        }
        self.image_area = image_width * image_height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, results) -> dict:
        """Extract cap-casting-specific attributes from YOLO results.

        Args:
            results: List of YOLOv8 Results objects returned by
                ``model.predict()``.

        Returns:
            dict: Named attributes for the rule engine, including counts,
                proximity flags, and density.
        """

        counts = {
            CLASS_CASTED_CAP: 0,
            CLASS_CAP_CASTING_CONCRETE_PUMP: 0,
            CLASS_CAP_FORMWORK: 0,
            CLASS_CAP_CASTING_VIBRATOR: 0,
            CLASS_PIER_STEM: 0,
            CLASS_CAP_CASTING_TRANSIT_MIXER: 0,
            CLASS_CAP_CASTING_WORKER: 0,
        }

        formwork_boxes: list = []
        casted_cap_boxes: list = []
        pier_stem_boxes: list = []
        worker_boxes: list = []
        casted_cap_area: float = 0.0

        # ------------------------------------------------------------------
        # Read all detections
        # ------------------------------------------------------------------

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                cls_name = self.class_names.get(cls, "").lower()
                xyxy = box.xyxy[0].tolist()

                if cls_name in counts:
                    counts[cls_name] += 1

                if cls_name == CLASS_CAP_FORMWORK:
                    formwork_boxes.append(xyxy)

                if cls_name == CLASS_CASTED_CAP:
                    casted_cap_boxes.append(xyxy)
                    casted_cap_area += box_area(xyxy)

                if cls_name == CLASS_PIER_STEM:
                    pier_stem_boxes.append(xyxy)

                if cls_name == CLASS_CAP_CASTING_WORKER:
                    worker_boxes.append(xyxy)

        # ------------------------------------------------------------------
        # Concrete Density
        # casted_cap pixel area as a fraction of the image area — used as
        # the concrete-presence proxy since this dataset has no
        # "fresh concrete" class (unlike Stage 2).
        # ------------------------------------------------------------------

        concrete_density = min(
            casted_cap_area / self.image_area, 1.0
        )

        # ------------------------------------------------------------------
        # Proximity flags — reuse shared helper exactly as used by
        # Stage 1, Stage 2, and Stage 3.
        # ------------------------------------------------------------------

        worker_near_formwork: bool = any_worker_near(
            worker_boxes, formwork_boxes
        )

        worker_near_casted_cap: bool = any_worker_near(
            worker_boxes, casted_cap_boxes
        )

        # ------------------------------------------------------------------
        # Cap-Stem Structural Relationship — same geometric disambiguation
        # as Stage 3, evaluated against cap_formwork and casted_cap boxes
        # combined (either can represent the cap-side object here).
        # ------------------------------------------------------------------

        cap_stem_relationship = evaluate_cap_stem_relationship(
            formwork_boxes + casted_cap_boxes,
            pier_stem_boxes,
        )

        # ------------------------------------------------------------------
        # Final Attributes
        # ------------------------------------------------------------------

        attributes = {

            "cap_formwork_count":
                counts[CLASS_CAP_FORMWORK],

            "concrete_pump_count":
                counts[CLASS_CAP_CASTING_CONCRETE_PUMP],

            "transit_mixer_count":
                counts[CLASS_CAP_CASTING_TRANSIT_MIXER],

            "vibrator_count":
                counts[CLASS_CAP_CASTING_VIBRATOR],

            "casted_cap_count":
                counts[CLASS_CASTED_CAP],

            "pier_stem_count":
                counts[CLASS_PIER_STEM],

            "worker_count":
                counts[CLASS_CAP_CASTING_WORKER],

            "concrete_density":
                round(concrete_density, 4),

            "worker_near_formwork":
                worker_near_formwork,

            "worker_near_casted_cap":
                worker_near_casted_cap,

            "cap_overhang_ratio":
                cap_stem_relationship["cap_overhang_ratio"],

            "cap_wider_than_stem":
                cap_stem_relationship["cap_wider_than_stem"],

            "cap_above_stem":
                cap_stem_relationship["cap_above_stem"],

            "cap_horizontally_aligned_with_stem":
                cap_stem_relationship["cap_horizontally_aligned_with_stem"],

            "cap_stem_relationship_confirmed":
                cap_stem_relationship["cap_stem_relationship_confirmed"],

            "total_objects":
                sum(counts.values()),
        }

        logger.info("========== CAP CASTING ATTRIBUTES ==========")
        logger.info(attributes)

        return attributes