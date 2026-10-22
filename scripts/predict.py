"""Batch inference CLI, using the same implementation as the workbench."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from pcb_inspector import ROOT
from pcb_inspector.dataset import read_image
from pcb_inspector.vision import Detector, draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/predictions")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()
    threshold_path = ROOT / "reports/threshold_selection.json"
    confidence = args.confidence if args.confidence is not None else json.loads(threshold_path.read_text())["chosen"] if threshold_path.exists() else .25
    if not 0 <= confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if args.source.is_dir() else [args.source]
    if not files:
        parser.error("No supported images found")
    args.output.mkdir(parents=True, exist_ok=True)
    detector = Detector()
    card = json.loads((ROOT / "models/model_card.json").read_text())
    rows = []
    for path in files:
        image = read_image(path)
        pred, info = detector.predict(image, confidence=confidence, tiled=args.tiled, imgsz=card["config"]["imgsz"])
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        name = path.stem + "_" + digest[:8] + ".png"
        cv2.imencode(".png", draw(image, pred))[1].tofile(str(args.output / name))
        rows.append({"image": path.name, "image_sha256": digest, "model_sha256": card["sha256"], "detections": [d.to_dict() for d in pred], "timing": info, "annotated_image": name})
    (args.output / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Processed {len(rows)} images → {args.output}")


if __name__ == "__main__":
    main()
