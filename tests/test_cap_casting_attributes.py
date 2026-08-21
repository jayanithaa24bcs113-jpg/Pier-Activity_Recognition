"""
Test Cap Casting Attributes

Purpose: Unit tests for CapCastingAttributeExtractor (Stage 4).
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Unit tests for CapCastingAttributeExtractor (Stage 4).

Mocks YOLOv8 ``results`` objects using lightweight stand-ins for
``result.boxes`` / ``box.cls`` / ``box.conf`` / ``box.xyxy`` so the
extractor can be tested without a real model or image.
"""

import pytest

from src.attributes.attribute_extractor import (
    CapCastingAttributeExtractor,
)

# ---------------------------------------------------------------------
# Class map — matches config.yaml activities.cap_casting.classes
# ---------------------------------------------------------------------

CLASS_NAMES = {
    0: "Casted cap",
    1: "Concrete pump",
    2: "cap formwork",
    3: "needle vibrator",
    4: "pier stem",
    5: "transit mixer",
    6: "worker",
}


# ---------------------------------------------------------------------
# Mock helpers — mimic the minimal YOLOv8 Results/Boxes interface
# used by CapCastingAttributeExtractor.extract()
# ---------------------------------------------------------------------

class _MockTensor:
    """Minimal stand-in for a single-element torch tensor.

    Indexing into a scalar list returns the raw scalar (so
    ``int()``/``float()`` work directly); indexing into a nested list
    returns another ``_MockTensor`` so ``.tolist()`` is available,
    matching ``box.xyxy[0].tolist()``.
    """

    def __init__(self, value):
        self._value = value

    def __getitem__(self, idx):
        item = self._value[idx]
        if isinstance(item, list):
            return _MockTensor(item)
        return item

    def tolist(self):
        return self._value

    def item(self):
        return self._value


class _MockBox:
    """Minimal stand-in for a single YOLOv8 detection box."""

    def __init__(self, cls_id: int, xyxy: list, conf: float = 0.9):
        self.cls = _MockTensor([cls_id])
        self.conf = _MockTensor([conf])
        self.xyxy = _MockTensor([xyxy])


class _MockResult:
    """Minimal stand-in for a single YOLOv8 Results object."""

    def __init__(self, boxes: list):
        self.boxes = boxes


def make_results(detections: list) -> list:
    """Build a mock ``results`` list from (class_id, xyxy) tuples.

    Args:
        detections: List of ``(class_id, [x1, y1, x2, y2])`` tuples, or
            ``(class_id, [x1, y1, x2, y2], conf)`` to override confidence.

    Returns:
        list[_MockResult]: A single-result list mimicking
            ``model.predict()``'s return value.
    """
    boxes = []
    for det in detections:
        if len(det) == 3:
            cls_id, xyxy, conf = det
        else:
            cls_id, xyxy = det
            conf = 0.9
        boxes.append(_MockBox(cls_id, xyxy, conf))
    return [_MockResult(boxes)]


# ---------------------------------------------------------------------
# Class index reference (per CLASS_NAMES above)
# 0: Casted cap, 1: Concrete pump, 2: cap formwork,
# 3: needle vibrator, 4: pier stem, 5: transit mixer, 6: worker
# ---------------------------------------------------------------------

CASTED_CAP = 0
CONCRETE_PUMP = 1
CAP_FORMWORK = 2
NEEDLE_VIBRATOR = 3
PIER_STEM = 4
TRANSIT_MIXER = 5
WORKER = 6


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_all_counts_present_and_correct():
    """Every documented attribute key must be present, with correct counts."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (CAP_FORMWORK, [0, 0, 300, 100]),
        (CONCRETE_PUMP, [0, 0, 50, 50]),
        (TRANSIT_MIXER, [0, 0, 60, 60]),
        (NEEDLE_VIBRATOR, [0, 0, 20, 40]),
        (CASTED_CAP, [0, 0, 400, 200]),
        (PIER_STEM, [0, 0, 100, 300]),
        (WORKER, [10, 10, 40, 120]),
    ])

    attrs = extractor.extract(results)

    expected_keys = {
        "cap_formwork_count",
        "concrete_pump_count",
        "transit_mixer_count",
        "vibrator_count",
        "casted_cap_count",
        "pier_stem_count",
        "worker_count",
        "concrete_density",
        "worker_near_formwork",
        "worker_near_casted_cap",
        "total_objects",
    }
    assert expected_keys.issubset(attrs.keys())

    assert attrs["cap_formwork_count"] == 1
    assert attrs["concrete_pump_count"] == 1
    assert attrs["transit_mixer_count"] == 1
    assert attrs["vibrator_count"] == 1
    assert attrs["casted_cap_count"] == 1
    assert attrs["pier_stem_count"] == 1
    assert attrs["worker_count"] == 1
    assert attrs["total_objects"] == 7


def test_concrete_density_uses_casted_cap_area():
    """concrete_density is casted_cap pixel area / image area (640x640)."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES, image_width=640, image_height=640)

    # casted_cap box: 400 x 200 = 80,000 px^2. image area = 409,600.
    results = make_results([
        (CASTED_CAP, [0, 0, 400, 200]),
    ])

    attrs = extractor.extract(results)

    expected_density = 80000 / (640 * 640)
    assert attrs["concrete_density"] == pytest.approx(expected_density, abs=0.0001)


def test_concrete_density_zero_with_no_casted_cap():
    """No casted_cap detections -> concrete_density == 0.0, no crash."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (WORKER, [10, 10, 40, 120]),
        (CAP_FORMWORK, [0, 0, 300, 100]),
    ])

    attrs = extractor.extract(results)

    assert attrs["concrete_density"] == 0.0


def test_concrete_density_capped_at_one():
    """An oversized casted_cap box must not push density above 1.0."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES, image_width=100, image_height=100)

    # Deliberately larger than the image area (100x100=10,000 px^2)
    results = make_results([
        (CASTED_CAP, [0, 0, 500, 500]),
    ])

    attrs = extractor.extract(results)

    assert attrs["concrete_density"] == 1.0


def test_worker_near_formwork_true():
    """A worker box near a cap formwork box should set worker_near_formwork True."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (CAP_FORMWORK, [100, 100, 300, 140]),   # centroid ~(200, 120)
        (WORKER, [180, 90, 220, 200]),          # centroid ~(200, 145) -> close
    ])

    attrs = extractor.extract(results)

    assert attrs["worker_near_formwork"] is True


def test_worker_near_formwork_false():
    """A worker box far from all formwork boxes should set worker_near_formwork False."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (CAP_FORMWORK, [0, 0, 100, 20]),
        (WORKER, [900, 900, 950, 1000]),
    ])

    attrs = extractor.extract(results)

    assert attrs["worker_near_formwork"] is False


def test_worker_near_casted_cap_true():
    """A worker box near a casted_cap box should set worker_near_casted_cap True."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (CASTED_CAP, [100, 100, 400, 200]),     # centroid ~(250, 150)
        (WORKER, [230, 120, 270, 220]),          # centroid ~(250, 170) -> close
    ])

    attrs = extractor.extract(results)

    assert attrs["worker_near_casted_cap"] is True


def test_worker_near_casted_cap_false():
    """A worker box far from all casted_cap boxes should set worker_near_casted_cap False."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (CASTED_CAP, [0, 0, 100, 50]),
        (WORKER, [900, 900, 950, 1000]),
    ])

    attrs = extractor.extract(results)

    assert attrs["worker_near_casted_cap"] is False


def test_empty_detections_returns_zeros_gracefully():
    """No detections at all should return all-zero/False attributes, no crash."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([])

    attrs = extractor.extract(results)

    assert attrs["cap_formwork_count"] == 0
    assert attrs["concrete_pump_count"] == 0
    assert attrs["transit_mixer_count"] == 0
    assert attrs["vibrator_count"] == 0
    assert attrs["casted_cap_count"] == 0
    assert attrs["pier_stem_count"] == 0
    assert attrs["worker_count"] == 0
    assert attrs["concrete_density"] == 0.0
    assert attrs["worker_near_formwork"] is False
    assert attrs["worker_near_casted_cap"] is False
    assert attrs["total_objects"] == 0


def test_multiple_detections_per_class_counted_correctly():
    """Repeated detections of the same class should all be counted."""
    extractor = CapCastingAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (CAP_FORMWORK, [0, 0, 100, 50]),
        (CAP_FORMWORK, [200, 200, 300, 250]),
        (WORKER, [10, 10, 40, 120]),
        (WORKER, [500, 500, 540, 620]),
        (WORKER, [50, 50, 90, 170]),
    ])

    attrs = extractor.extract(results)

    assert attrs["cap_formwork_count"] == 2
    assert attrs["worker_count"] == 3
    assert attrs["total_objects"] == 5