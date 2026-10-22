"""Integration checks against the trained local artifact (no mocked model)."""
import json
import math
import pytest
from pcb_inspector import ROOT
from pcb_inspector.dataset import read_image
from pcb_inspector.vision import Detector


@pytest.mark.skipif(not (ROOT / "models/pcb_yolo11n.pt").exists() or not (ROOT / "data/processed/manifest.json").exists(), reason="Trained model and prepared dataset required")
def test_real_model_full_and_tiled_coordinate_bounds():
    records = json.loads((ROOT / "data/processed/manifest.json").read_text())["records"]
    record = next(r for r in records if r["split"] == "test")
    image = read_image(ROOT / record["image"])
    detector = Detector()
    for tiled in [False, True]:
        predictions, timing = detector.predict(image, confidence=.05, tiled=tiled)
        assert predictions, "The held-out demo image should produce candidates at low confidence"
        assert timing["tiles"] == (9 if tiled else 1)
        for prediction in predictions:
            x1, y1, x2, y2 = prediction.box
            assert all(math.isfinite(v) for v in prediction.box)
            assert 0 <= x1 < x2 <= image.shape[1]
            assert 0 <= y1 < y2 <= image.shape[0]
            assert 0 <= prediction.class_id < 6
            assert .05 <= prediction.confidence <= 1
