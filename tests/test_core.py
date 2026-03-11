"""
Unit tests for AIT Visual Inspector core modules.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests for src.utils.metrics
# ---------------------------------------------------------------------------

class TestComputeIoU:
    """Tests for compute_iou function."""

    def test_identical_boxes(self):
        from src.utils.metrics import compute_iou
        box = np.array([10, 10, 50, 50])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.utils.metrics import compute_iou
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from src.utils.metrics import compute_iou
        box1 = np.array([0, 0, 20, 20])
        box2 = np.array([10, 10, 30, 30])
        assert compute_iou(box1, box2) == pytest.approx(100.0 / 700.0, abs=0.01)

    def test_contained_box(self):
        from src.utils.metrics import compute_iou
        box1 = np.array([0, 0, 40, 40])
        box2 = np.array([10, 10, 20, 20])
        assert compute_iou(box1, box2) == pytest.approx(100.0 / 1600.0, abs=0.01)

    def test_zero_area_box(self):
        from src.utils.metrics import compute_iou
        box1 = np.array([10, 10, 10, 10])
        box2 = np.array([0, 0, 20, 20])
        assert compute_iou(box1, box2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests for detector verdict logic (multi-class)
# ---------------------------------------------------------------------------

class TestDetermineVerdict:
    """Tests for CircuitSight_Detector._determine_verdict with multi-class PCB defects."""

    def _make_detector(self, pass_threshold=0.5, review_threshold=0.3, max_defects_pass=0):
        from src.models.detector import CircuitSight_Detector
        d = CircuitSight_Detector(
            pass_threshold=pass_threshold,
            review_threshold=review_threshold,
            max_defects_pass=max_defects_pass,
        )
        return d

    def _make_detection(self, confidence, class_name="short_circuit", class_id=3):
        from src.models.detector import Detection
        return Detection(
            class_id=class_id, class_name=class_name, confidence=confidence,
            bbox=[0, 0, 10, 10], bbox_norm=[0.5, 0.5, 0.1, 0.1],
        )

    def test_no_detections_is_pass(self):
        d = self._make_detector()
        assert d._determine_verdict([]) == "PASS"

    def test_high_confidence_defect_is_fail(self):
        d = self._make_detector(pass_threshold=0.5, max_defects_pass=0)
        det = self._make_detection(0.8, "missing_hole", 0)
        assert d._determine_verdict([det]) == "FAIL"

    def test_mid_confidence_defect_is_review(self):
        d = self._make_detector(pass_threshold=0.5, review_threshold=0.3, max_defects_pass=0)
        det = self._make_detection(0.4, "mouse_bite", 1)
        assert d._determine_verdict([det]) == "NEEDS_REVIEW"

    def test_low_confidence_defect_is_pass(self):
        d = self._make_detector(pass_threshold=0.5, review_threshold=0.3, max_defects_pass=0)
        det = self._make_detection(0.2, "spur", 4)
        assert d._determine_verdict([det]) == "PASS"

    def test_multiple_defect_types(self):
        d = self._make_detector(pass_threshold=0.5, max_defects_pass=0)
        dets = [
            self._make_detection(0.8, "missing_hole", 0),
            self._make_detection(0.7, "open_circuit", 2),
        ]
        assert d._determine_verdict(dets) == "FAIL"

    def test_max_defects_pass_allows_some(self):
        d = self._make_detector(pass_threshold=0.5, max_defects_pass=2)
        dets = [
            self._make_detection(0.8, "short_circuit", 3),
            self._make_detection(0.7, "spur", 4),
        ]
        assert d._determine_verdict(dets) == "PASS"

    def test_max_defects_pass_exceeded(self):
        d = self._make_detector(pass_threshold=0.5, max_defects_pass=1)
        dets = [
            self._make_detection(0.8, "short_circuit", 3),
            self._make_detection(0.7, "spurious_copper", 5),
        ]
        assert d._determine_verdict(dets) == "FAIL"

    def test_boundary_at_pass_threshold(self):
        d = self._make_detector(pass_threshold=0.5, max_defects_pass=0)
        det = self._make_detection(0.5, "open_circuit", 2)
        assert d._determine_verdict([det]) == "FAIL"

    def test_boundary_at_review_threshold(self):
        d = self._make_detector(pass_threshold=0.5, review_threshold=0.3, max_defects_pass=0)
        det = self._make_detection(0.3, "mouse_bite", 1)
        assert d._determine_verdict([det]) == "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Tests for data augmentation
# ---------------------------------------------------------------------------

class TestAugmentImage:
    """Tests for augment_image function."""

    def test_output_shape_matches_input(self):
        from src.data.augment import augment_image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result, bboxes = augment_image(img)
        assert result.shape == img.shape

    def test_returns_at_least_one_bbox(self):
        from src.data.augment import augment_image
        import random
        random.seed(42)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result, bboxes = augment_image(img)
        assert len(bboxes) >= 1

    def test_bbox_values_normalized(self):
        from src.data.augment import augment_image
        import random
        random.seed(42)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result, bboxes = augment_image(img)
        for bbox in bboxes:
            assert bbox[0] == 1  # class_id = 1 (defect)
            assert 0 <= bbox[1] <= 1.1
            assert 0 <= bbox[2] <= 1.1

    def test_does_not_modify_original(self):
        from src.data.augment import augment_image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = img.copy()
        augment_image(img)
        np.testing.assert_array_equal(img, original)


# ---------------------------------------------------------------------------
# Tests for CLAHE preprocessing
# ---------------------------------------------------------------------------

class TestCLAHE:
    """Tests for CLAHE contrast enhancement."""

    def test_clahe_output_shape(self):
        from src.data.augment import apply_clahe
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_clahe(img)
        assert result.shape == img.shape

    def test_clahe_output_dtype(self):
        from src.data.augment import apply_clahe
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_clahe(img)
        assert result.dtype == np.uint8

    def test_clahe_does_not_modify_original(self):
        from src.data.augment import apply_clahe
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = img.copy()
        apply_clahe(img)
        np.testing.assert_array_equal(img, original)


# ---------------------------------------------------------------------------
# Tests for VOC XML parsing
# ---------------------------------------------------------------------------

class TestVOCParser:
    """Tests for parse_voc_xml function."""

    def test_parse_valid_xml(self, tmp_path):
        from src.data.convert_to_yolo import parse_voc_xml
        xml_content = """<?xml version="1.0"?>
        <annotation>
            <object>
                <name>missing_hole</name>
                <bndbox>
                    <xmin>10</xmin><ymin>20</ymin>
                    <xmax>50</xmax><ymax>60</ymax>
                </bndbox>
            </object>
            <object>
                <name>short_circuit</name>
                <bndbox>
                    <xmin>100</xmin><ymin>100</ymin>
                    <xmax>200</xmax><ymax>200</ymax>
                </bndbox>
            </object>
        </annotation>"""
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)

        objects = parse_voc_xml(str(xml_path))
        assert len(objects) == 2
        assert objects[0][0] == "missing_hole"
        assert objects[0][1] == 10  # x1
        assert objects[1][0] == "short_circuit"

    def test_parse_empty_xml(self, tmp_path):
        from src.data.convert_to_yolo import parse_voc_xml
        xml_content = '<?xml version="1.0"?><annotation></annotation>'
        xml_path = tmp_path / "empty.xml"
        xml_path.write_text(xml_content)

        objects = parse_voc_xml(str(xml_path))
        assert len(objects) == 0

    def test_nonexistent_xml(self):
        from src.data.convert_to_yolo import parse_voc_xml
        objects = parse_voc_xml("/nonexistent/path.xml")
        assert len(objects) == 0


# ---------------------------------------------------------------------------
# Tests for VOC to YOLO coordinate conversion
# ---------------------------------------------------------------------------

class TestVocToYolo:
    """Tests for voc_to_yolo_bbox conversion."""

    def test_center_of_image(self):
        from src.data.convert_to_yolo import voc_to_yolo_bbox
        cx, cy, w, h = voc_to_yolo_bbox(100, 100, 300, 300, 400, 400)
        assert cx == pytest.approx(0.5, abs=0.01)
        assert cy == pytest.approx(0.5, abs=0.01)
        assert w == pytest.approx(0.5, abs=0.01)
        assert h == pytest.approx(0.5, abs=0.01)

    def test_top_left_corner(self):
        from src.data.convert_to_yolo import voc_to_yolo_bbox
        cx, cy, w, h = voc_to_yolo_bbox(0, 0, 100, 100, 400, 400)
        assert cx == pytest.approx(0.125, abs=0.01)
        assert cy == pytest.approx(0.125, abs=0.01)
        assert w == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------------------
# Tests for mask_to_bboxes (MVTec legacy)
# ---------------------------------------------------------------------------

class TestMaskToBboxes:
    """Tests for mask_to_bboxes function."""

    def test_empty_mask(self, tmp_path):
        import cv2
        from src.data.convert_to_yolo import mask_to_bboxes
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask_path = str(tmp_path / "empty.png")
        cv2.imwrite(mask_path, mask)
        assert mask_to_bboxes(mask_path) == []

    def test_single_contour(self, tmp_path):
        import cv2
        from src.data.convert_to_yolo import mask_to_bboxes
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
        mask_path = str(tmp_path / "single.png")
        cv2.imwrite(mask_path, mask)

        bboxes = mask_to_bboxes(mask_path)
        assert len(bboxes) == 1
        assert bboxes[0][0] == 1
        assert 0.4 <= bboxes[0][1] <= 0.6
        assert 0.4 <= bboxes[0][2] <= 0.6

    def test_nonexistent_file(self):
        from src.data.convert_to_yolo import mask_to_bboxes
        assert mask_to_bboxes("/nonexistent/path.png") == []


# ---------------------------------------------------------------------------
# Tests for QC report generation
# ---------------------------------------------------------------------------

class TestGenerateQCReport:
    """Tests for generate_qc_report function."""

    def _make_result(self, verdict="PASS", n_defects=0):
        from src.models.detector import InferenceResult, Detection
        detections = []
        defect_names = ["missing_hole", "mouse_bite", "open_circuit",
                        "short_circuit", "spur", "spurious_copper"]
        for i in range(n_defects):
            detections.append(Detection(
                class_id=i % 6, class_name=defect_names[i % 6], confidence=0.8,
                bbox=[0, 0, 10, 10], bbox_norm=[0.5, 0.5, 0.1, 0.1],
            ))
        return InferenceResult(
            image_path=f"test_{n_defects}.png",
            image_shape=(100, 100, 3),
            detections=detections,
            inference_time_ms=25.0,
            verdict=verdict,
        )

    def test_empty_results(self):
        from src.reporting.qc_report import generate_qc_report
        report = generate_qc_report([])
        assert report["n_inspected"] == 0
        assert report["summary"]["pass"] == 0

    def test_all_pass(self):
        from src.reporting.qc_report import generate_qc_report
        results = [self._make_result("PASS") for _ in range(5)]
        report = generate_qc_report(results, batch_name="test_batch")
        assert report["n_inspected"] == 5
        assert report["summary"]["pass"] == 5
        assert report["summary"]["fail"] == 0
        assert report["summary"]["pass_rate"] == 100.0

    def test_mixed_verdicts(self):
        from src.reporting.qc_report import generate_qc_report
        results = [
            self._make_result("PASS"),
            self._make_result("FAIL", n_defects=2),
            self._make_result("NEEDS_REVIEW", n_defects=1),
        ]
        report = generate_qc_report(results)
        assert report["summary"]["pass"] == 1
        assert report["summary"]["fail"] == 1
        assert report["summary"]["needs_review"] == 1
        assert report["summary"]["total_defects"] == 3


# ---------------------------------------------------------------------------
# Tests for compute_detection_metrics
# ---------------------------------------------------------------------------

class TestComputeDetectionMetrics:
    """Tests for compute_detection_metrics."""

    def test_perfect_detection(self):
        from src.utils.metrics import compute_detection_metrics
        preds = [{"boxes": [[10, 10, 50, 50]], "scores": [0.9], "classes": [1]}]
        gts = [{"boxes": [[10, 10, 50, 50]], "classes": [1]}]
        metrics = compute_detection_metrics(preds, gts, iou_threshold=0.5, n_classes=2)
        assert metrics["class_1"]["tp"] == 1
        assert metrics["class_1"]["fp"] == 0
        assert metrics["class_1"]["fn"] == 0

    def test_missed_detection(self):
        from src.utils.metrics import compute_detection_metrics
        preds = [{"boxes": [], "scores": [], "classes": []}]
        gts = [{"boxes": [[10, 10, 50, 50]], "classes": [1]}]
        metrics = compute_detection_metrics(preds, gts, iou_threshold=0.5, n_classes=2)
        assert metrics["class_1"]["fn"] == 1

    def test_false_positive(self):
        from src.utils.metrics import compute_detection_metrics
        preds = [{"boxes": [[10, 10, 50, 50]], "scores": [0.9], "classes": [1]}]
        gts = [{"boxes": [], "classes": []}]
        metrics = compute_detection_metrics(preds, gts, iou_threshold=0.5, n_classes=2)
        assert metrics["class_1"]["fp"] == 1

    def test_empty_inputs(self):
        from src.utils.metrics import compute_detection_metrics
        metrics = compute_detection_metrics([], [], iou_threshold=0.5, n_classes=2)
        assert metrics["overall"]["precision"] == 0.0
