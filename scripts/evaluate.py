"""Select deployment confidence on validation ONLY; evaluate untouched test split.

Class-agnostic localization is reported separately so OpenCV's unclassified
regions are not misleadingly compared against six-class YOLO classification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pcb_inspector import ROOT
from pcb_inspector.dataset import read_image
from pcb_inspector.metrics import match, summarize
from pcb_inspector.vision import Detector, draw, template_difference
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/processed/manifest.json").read_text())
    baseline_parameters = json.loads((ROOT / "reports/baseline_selection.json").read_text())["chosen"]
    detector = Detector()
    detector.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8), imgsz=args.imgsz)
    val = [r for r in manifest["records"] if r["split"] == "val"]
    validation = []
    for record in val:
        pred, _ = detector.predict(read_image(ROOT / record["image"]), confidence=0.01, imgsz=args.imgsz)
        validation.append((pred, record["boxes"]))
    sweep = []
    for confidence in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for pred, gt in validation:
            m = match([p for p in pred if p.confidence >= confidence], gt)
            for k in totals:
                totals[k] += m[k]
        sweep.append({"confidence": confidence, **summarize(**totals)})
    chosen = max(sweep, key=lambda s: (s["f1"], s["confidence"]))["confidence"]
    (ROOT / "reports/threshold_selection.json").write_text(json.dumps({"split": "val", "objective": "six-class micro F1 at IoU 0.5", "chosen": chosen, "sweep": sweep}, indent=2))
    test = [r for r in manifest["records"] if r["split"] == "test"]
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ["yolo_class_aware", "yolo_localization", "opencv_localization"]}
    latency = {"yolo": [], "opencv": []}
    per_image = []
    failure_dir = ROOT / "reports/failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failures_saved = 0
    for index, record in enumerate(test):
        image = read_image(ROOT / record["image"])
        pred, timing = detector.predict(image, confidence=chosen, imgsz=args.imgsz)
        baseline, _, bt = template_difference(image, read_image(ROOT / record["template"]), **baseline_parameters)
        matches = {"yolo_class_aware": match(pred, record["boxes"]), "yolo_localization": match(pred, record["boxes"], class_aware=False), "opencv_localization": match(baseline, record["boxes"], class_aware=False)}
        for name, result in matches.items():
            for key in totals[name]:
                totals[name][key] += result[key]
        latency["yolo"].append(timing["elapsed_ms"])
        latency["opencv"].append(bt["elapsed_ms"])
        per_image.append({"id": record["id"], "matches": matches, "yolo": [p.to_dict() for p in pred], "opencv": [p.to_dict() for p in baseline], "latency_ms": {"yolo": timing["elapsed_ms"], "opencv": bt["elapsed_ms"]}})
        if (matches["yolo_class_aware"]["fp"] or matches["yolo_class_aware"]["fn"]) and failures_saved < 12:
            cv2.imencode(".png", draw(image, pred))[1].tofile(str(failure_dir / f'{record["id"]}.png'))
            failures_saved += 1
        if index % 25 == 0:
            print(f"Test {index+1}/{len(test)}", flush=True)
    # Ultralytics mAP has its own low confidence floor; deployment threshold is separate.
    metrics = detector.model.val(data=str(ROOT / "data/processed/data.yaml"), split="test", imgsz=args.imgsz, batch=8, device="cpu", workers=0, conf=0.001, iou=0.45, plots=True, project=str(ROOT / "runs"), name="test_evaluation", verbose=False)
    report = {"split": "test", "n_images": len(test), "confidence": chosen, "matching_iou": 0.5, "imgsz": args.imgsz, "device": "cpu", "model_sha256": hashlib.sha256((ROOT / "models/pcb_yolo11n.pt").read_bytes()).hexdigest(), "metrics": {name: summarize(**counts) for name, counts in totals.items()}, "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class_map50_95": [float(v) for v in metrics.box.maps], "latency_ms": {name: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "scope": "in-memory image to detections; excludes disk read, model load and drawing; CPU warmup performed"} for name, values in latency.items()}, "protocol": manifest["protocol"], "notes": ["OpenCV requires aligned defect-free template; YOLO consumes only tested image", "OpenCV has no defect classification; compare localization columns only", "Validation selects confidence. Test metrics are descriptive, not a production guarantee", "Timed on this machine with possible concurrent workloads; not a controlled hardware benchmark"]}
    report["baseline_parameters"] = baseline_parameters
    (ROOT / "reports/evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "reports/per_image_results.json").write_text(json.dumps(per_image, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
