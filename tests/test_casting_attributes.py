"""Unit tests for CastingAttributeExtractor.

These tests validate attribute extraction for Pier Stem Casting using
mock YOLO result objects, mirroring the fixture style used for the
reinforcement rule engine tests.
"""

import unittest
from unittest.mock import MagicMock

from src.attributes.attribute_extractor import CastingAttributeExtractor


CLASS_NAMES = {
    0: "Formwork",
    1: "Concrete Pump",
    2: "Transit Mixer",
    3: "Vibrator",
    4: "Fresh Concrete",
    5: "Worker",
    6: "Casted Pier",
}


def _make_box(cls_id: int, xyxy: list) -> MagicMock:
    """Create a mock YOLO box object.

    Args:
        cls_id: Class index for this box.
        xyxy: [x1, y1, x2, y2] coordinates.

    Returns:
        MagicMock: Mock box exposing ``.cls`` and ``.xyxy`` like a real
            YOLOv8 Boxes entry.
    """
    box = MagicMock()
    box.cls = [cls_id]
    box.xyxy = [MagicMock(tolist=lambda: xyxy)]
    return box


def _make_results(boxes: list) -> list:
    """Wrap a list of mock boxes into a mock YOLO Results list.

    Args:
        boxes: List of mock box objects.

    Returns:
        list: Single-element list containing a mock Result with
            ``.boxes`` set to ``boxes``.
    """
    result = MagicMock()
    result.boxes = boxes
    return [result]


class TestCastingAttributeExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = CastingAttributeExtractor(
            class_names=CLASS_NAMES,
            image_width=640,
            image_height=640,
        )

    def test_empty_results_returns_zero_counts(self):
        results = _make_results([])
        attrs = self.extractor.extract(results)
        self.assertEqual(attrs["formwork_count"], 0)
        self.assertEqual(attrs["concrete_pump_count"], 0)
        self.assertEqual(attrs["total_objects"], 0)

    def test_formwork_count(self):
        boxes = [
            _make_box(0, [0, 0, 50, 100]),
            _make_box(0, [100, 0, 150, 100]),
        ]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertEqual(attrs["formwork_count"], 2)

    def test_worker_near_formwork_true(self):
        boxes = [
            _make_box(0, [0, 0, 50, 100]),      # formwork
            _make_box(5, [20, 20, 70, 120]),    # worker, close by
        ]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertTrue(attrs["worker_near_formwork"])

    def test_worker_near_formwork_false_when_far(self):
        boxes = [
            _make_box(0, [0, 0, 50, 100]),          # formwork
            _make_box(5, [2000, 2000, 2050, 2100]), # worker, far away
        ]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertFalse(attrs["worker_near_formwork"])

    def test_worker_near_concrete_true(self):
        boxes = [
            _make_box(4, [10, 10, 60, 60]),     # fresh concrete
            _make_box(5, [20, 20, 70, 70]),      # worker, close by
        ]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertTrue(attrs["worker_near_concrete"])

    def test_concrete_density_calculation(self):
        # fresh concrete box covers 100x100 = 10000 px out of 640*640 = 409600
        boxes = [_make_box(4, [0, 0, 100, 100])]
        attrs = self.extractor.extract(_make_results(boxes))
        expected_density = round(10000 / (640 * 640), 4)
        self.assertAlmostEqual(attrs["concrete_density"], expected_density)

    def test_tall_vertical_shuttering_ratio(self):
        # formwork box: width=50, height=200 - ratio = 4.0
        boxes = [_make_box(0, [0, 0, 50, 200])]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertAlmostEqual(attrs["tall_vertical_shuttering"], 4.0)

    def test_tall_vertical_shuttering_zero_when_no_formwork(self):
        boxes = [_make_box(5, [0, 0, 50, 100])]  # worker only
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertEqual(attrs["tall_vertical_shuttering"], 0.0)

    def test_casted_pier_count(self):
        boxes = [_make_box(6, [0, 0, 50, 50])]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertEqual(attrs["casted_pier_count"], 1)

    def test_total_objects_sums_all_counts(self):
        boxes = [
            _make_box(0, [0, 0, 10, 10]),   # formwork
            _make_box(1, [0, 0, 10, 10]),   # concrete pump
            _make_box(5, [0, 0, 10, 10]),   # worker
            _make_box(5, [0, 0, 10, 10]),   # worker
        ]
        attrs = self.extractor.extract(_make_results(boxes))
        self.assertEqual(attrs["total_objects"], 4)


if __name__ == "__main__":
    unittest.main()