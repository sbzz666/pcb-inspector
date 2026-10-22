"""Tune only the OpenCV baseline on validation, before looking at test metrics."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcb_inspector import ROOT
from pcb_inspector.dataset import read_image
from pcb_inspector.metrics import match, summarize
from pcb_inspector.vision import template_difference


def main():
    records = json.loads((ROOT / "data/processed/manifest.json").read_text())["records"]
    samples = [(read_image(ROOT/r["image"]), read_image(ROOT/r["template"]), r["boxes"]) for r in records if r["split"] == "val"]
    sweep = []
    for tolerance in [1, 2, 3]:
        for min_area in [8, 16, 32]:
            counts = {"tp": 0, "fp": 0, "fn": 0}
            for image, template, boxes in samples:
                pred, _, _ = template_difference(image, template, tolerance=tolerance, min_area=min_area)
                m = match(pred, boxes, class_aware=False)
                for k in counts:
                    counts[k] += m[k]
            sweep.append({"tolerance": tolerance, "min_area": min_area, **summarize(**counts)})
    best = max(sweep, key=lambda row: row["f1"])
    report = {"split": "val", "objective": "class-agnostic micro F1 at IoU 0.5", "chosen": {"tolerance": best["tolerance"], "min_area": best["min_area"]}, "sweep": sweep}
    (ROOT / "reports/baseline_selection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
