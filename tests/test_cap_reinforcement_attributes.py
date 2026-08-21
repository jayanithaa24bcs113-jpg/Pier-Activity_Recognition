"""
Test Cap Reinforcement Attributes

Purpose: Unit tests for CapReinforcementAttributeExtractor (Stage 3).
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Unit tests for CapReinforcementAttributeExtractor (Stage 3).

Mocks YOLOv8 ``results`` objects using lightweight stand-ins for
``result.boxes`` / ``box.cls`` / ``box.conf`` / ``box.xyxy`` so the
extractor can be tested without a real model or image.
"""

import pytest

from src.attributes.attribute_extractor import (
    CapReinforcementAttributeExtractor,
)

# ---------------------------------------------------------------------
# Class map — matches config.yaml activities.cap_reinforcement.classes
# ---------------------------------------------------------------------

CLASS_NAMES = {
    0: "Horizontal Rebar",
    1: "Pier stem",
    2: "rebar_cage",
    3: "Worker",
    4: "Crane",
}


# ---------------------------------------------------------------------
# Mock helpers — mimic the minimal YOLOv8 Results/Boxes interface
# used by CapReinforcementAttributeExtractor.extract()
# ---------------------------------------------------------------------

class _MockTensor:
    """Minimal stand-in for a single-element torch tensor.

    Supports ``[0]`` indexing the same way a real YOLO box tensor would
    when the extractor calls ``box.cls[0]``, ``box.conf[0]``, or
    ``box.xyxy[0].tolist()``. Indexing into a scalar list returns the
    raw scalar (so ``int()``/``float()`` work directly); indexing into
    a nested list returns another ``_MockTensor`` so ``.tolist()`` is
    available on the result, matching ``box.xyxy[0].tolist()``.
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
# Tests
# ---------------------------------------------------------------------

def test_avg_rebar_aspect_ratio_wide_bboxes():
    """Wide horizontal_rebar boxes should yield an aspect ratio > 4.0."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    # width=400, height=50 -> ratio = 8.0
    results = make_results([
        (0, [0, 0, 400, 50]),
        (0, [0, 100, 440, 160]),  # width=440, height=60 -> ratio ~7.33
    ])

    attrs = extractor.extract(results)

    assert attrs["horizontal_rebar_count"] == 2
    assert attrs["avg_rebar_aspect_ratio"] > 4.0


def test_avg_rebar_aspect_ratio_square_bboxes():
    """Square horizontal_rebar boxes should yield an aspect ratio ~1.0."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    # width=100, height=100 -> ratio = 1.0
    results = make_results([
        (0, [0, 0, 100, 100]),
    ])

    attrs = extractor.extract(results)

    assert attrs["avg_rebar_aspect_ratio"] == pytest.approx(1.0, abs=0.01)


def test_avg_cage_aspect_ratio_wide_bboxes():
    """Wide rebar_cage boxes should yield an aspect ratio > 2.0."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    # width=300, height=100 -> ratio = 3.0
    results = make_results([
        (2, [0, 0, 300, 100]),
    ])

    attrs = extractor.extract(results)

    assert attrs["rebar_cage_count"] == 1
    assert attrs["avg_cage_aspect_ratio"] > 2.0


def test_rebar_density_computation():
    """rebar_density = (horizontal_rebar + rebar_cage) / total_objects."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (0, [0, 0, 400, 50]),     # horizontal_rebar
        (2, [0, 0, 300, 100]),    # rebar_cage
        (1, [0, 0, 500, 500]),    # pier_stem_top
        (3, [10, 10, 40, 120]),   # worker
    ])

    attrs = extractor.extract(results)

    # 2 rebar-type detections out of 4 total objects
    assert attrs["total_objects"] == 4
    assert attrs["rebar_density"] == pytest.approx(0.5, abs=0.0001)


def test_rebar_density_zero_total_objects():
    """rebar_density must not raise ZeroDivisionError when there are no detections."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    results = make_results([])

    attrs = extractor.extract(results)

    assert attrs["total_objects"] == 0
    assert attrs["rebar_density"] == 0.0


def test_worker_near_rebar_proximity_true():
    """A worker box near a horizontal_rebar box should set worker_near_rebar True."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (0, [100, 100, 300, 140]),   # horizontal_rebar, centroid ~(200, 120)
        (3, [180, 90, 220, 200]),    # worker, centroid ~(200, 145) -> close
    ])

    attrs = extractor.extract(results)

    assert attrs["worker_near_rebar"] is True


def test_worker_near_rebar_proximity_false():
    """A worker box far from all rebar/cage boxes should set worker_near_rebar False."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (0, [0, 0, 100, 20]),          # horizontal_rebar near origin
        (3, [900, 900, 950, 1000]),    # worker far away
    ])

    attrs = extractor.extract(results)

    assert attrs["worker_near_rebar"] is False


def test_empty_detections_returns_zeros_gracefully():
    """No detections at all should return all-zero/False attributes, no crash."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    results = make_results([])

    attrs = extractor.extract(results)

    assert attrs["horizontal_rebar_count"] == 0
    assert attrs["pier_stem_top_count"] == 0
    assert attrs["rebar_cage_count"] == 0
    assert attrs["worker_count"] == 0
    assert attrs["crane_count"] == 0
    assert attrs["rebar_density"] == 0.0
    assert attrs["avg_rebar_aspect_ratio"] == 0.0
    assert attrs["avg_cage_aspect_ratio"] == 0.0
    assert attrs["worker_near_rebar"] is False
    assert attrs["total_objects"] == 0


def test_all_counts_present_in_output():
    """Every documented attribute key must be present in the output dict."""
    extractor = CapReinforcementAttributeExtractor(CLASS_NAMES)

    results = make_results([
        (0, [0, 0, 400, 50]),
        (1, [0, 0, 500, 500]),
        (2, [0, 0, 300, 100]),
        (3, [10, 10, 40, 120]),
        (4, [0, 0, 60, 400]),
    ])

    attrs = extractor.extract(results)

    expected_keys = {
        "horizontal_rebar_count",
        "pier_stem_top_count",
        "rebar_cage_count",
        "worker_count",
        "crane_count",
        "rebar_density",
        "avg_rebar_aspect_ratio",
        "avg_cage_aspect_ratio",
        "worker_near_rebar",
        "total_objects",
    }

    assert expected_keys.issubset(attrs.keys())
    assert attrs["crane_count"] == 1
    assert attrs["total_objects"] == 5