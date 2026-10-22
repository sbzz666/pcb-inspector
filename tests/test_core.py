import json
import cv2
import numpy as np
import pytest
from pcb_inspector.dataset import read_labels, yolo_line
from pcb_inspector.metrics import match, summarize
from pcb_inspector.review import list_reviews, save_review
from pcb_inspector.vision import Detection, iou, suppress, template_difference, tile_starts


def test_label_schema_and_normalization(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("10 20 30 60 6\n")
    boxes = read_labels(label, 100, 100)
    assert boxes[0]["class_id"] == 5
    assert yolo_line(boxes[0], 100, 100).split() == ["5", "0.20000000", "0.40000000", "0.20000000", "0.40000000"]
    label.write_text("-1 0 10 10 1")
    with pytest.raises(ValueError):
        read_labels(label, 100, 100)


def test_one_gt_cannot_match_two_detections():
    preds = [Detection([0, 0, 10, 10], 0, 0.9), Detection([0, 0, 10, 10], 0, 0.8)]
    gt = [{"box": [0, 0, 10, 10], "class_id": 0}]
    result = match(preds, gt)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)


def test_class_aware_and_localization_are_different():
    pred = [Detection([0, 0, 10, 10], 1, 0.9)]
    gt = [{"box": [0, 0, 10, 10], "class_id": 0}]
    assert match(pred, gt)["tp"] == 0
    assert match(pred, gt, class_aware=False)["tp"] == 1
    assert summarize(0, 0, 0)["f1"] == 0


def test_tile_coverage_including_last_pixel():
    starts = tile_starts(1000, 320, .25)
    covered = np.zeros(1000, dtype=bool)
    for s in starts:
        covered[s:s+320] = True
    assert covered.all()
    assert tile_starts(100, 320, .25) == [0]
    with pytest.raises(ValueError):
        tile_starts(100, 320, 1)


def test_class_aware_global_nms():
    boxes = [Detection([0, 0, 20, 20], 0, .9), Detection([1, 1, 20, 20], 0, .8), Detection([0, 0, 20, 20], 1, .7)]
    assert len(suppress(boxes)) == 2
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0


def test_template_difference_identical_and_added_defect():
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not template_difference(template, template)[0]
    image = template.copy()
    image[30:40, 40:50] = 255
    detections, mask, _ = template_difference(image, template)
    assert len(detections) == 1 and detections[0].class_id == -1
    assert iou(detections[0].box, [34, 24, 56, 46]) > .9
    with pytest.raises(ValueError):
        template_difference(image, template[:50])


def test_review_is_append_only_and_preserves_quotes(tmp_path):
    path = tmp_path / "review.sqlite3"
    payload = {"note": "断路 ' 确认", "confidence": .25}
    first = save_review("abc", "image.jpg", payload, path)
    second = save_review("abc", "image.jpg", {"note": "再次复核"}, path)
    rows = list_reviews(path)
    assert second > first and len(rows) == 2
    assert rows[1]["payload"] == payload


def test_ecc_recovers_translation():
    rng = np.random.default_rng(42)
    template = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    template = cv2.GaussianBlur(template, (9, 9), 0)
    shifted = cv2.warpAffine(template, np.array([[1, 0, 2], [0, 1, 3]], np.float32), (128, 128), borderMode=cv2.BORDER_REPLICATE)
    _, _, info = template_difference(shifted, template, align=True)
    assert abs(abs(info["registration_shift"][0])-2) < .5
    assert abs(abs(info["registration_shift"][1])-3) < .5
