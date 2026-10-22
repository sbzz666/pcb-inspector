"""Reproducible CPU/GPU training; writes a self-contained model card."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "local_config"))
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", default="pcb_yolo11n")
    parser.add_argument("--resume", type=Path, help="Resume a trusted local last.pt checkpoint")
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    initial = ROOT / "models/yolo11n-official.pt"
    if not initial.exists():
        initial = ROOT / "models/yolo11n.pt"
    model = YOLO(str(args.resume or initial))
    previous_seconds = 0.0
    if args.resume:
        prior_csv = args.resume.resolve().parents[1] / "results.csv"
        if prior_csv.exists():
            previous_rows = list(csv.DictReader(prior_csv.open(encoding="utf-8")))
            last_time = 0.0
            for row in previous_rows:
                current_time = float(row["time"])
                if current_time < last_time:
                    previous_seconds += last_time
                last_time = current_time
            previous_seconds += last_time
    started = time.perf_counter()
    if args.resume:
        model.train(resume=True, device=args.device, workers=0)
    else:
        model.train(data=str(ROOT / "data/processed/data.yaml"), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device, workers=0,
                project=str(ROOT / "runs"), name=args.name, seed=42, deterministic=True,
                optimizer="AdamW", lr0=0.001, patience=10, cache=False,
                mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
                translate=0.05, scale=0.1, fliplr=0.5, flipud=0.5,
                amp=False, plots=True, verbose=False)
    run = Path(model.trainer.save_dir)
    source = run / "weights/best.pt"
    shutil.copy2(source, ROOT / "models/pcb_yolo11n.pt")
    config = {**vars(args), "resume": str(args.resume) if args.resume else None}
    card = {"architecture": "YOLO11n", "pretrained": "Ultralytics COCO yolo11n.pt", "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": platform.python_version(), "device": args.device, "processor": platform.processor(), "threads": args.threads, "config": config, "run": run.relative_to(ROOT).as_posix(), "training_seconds": previous_seconds+time.perf_counter()-started, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "limitations": ["Public preprocessed DeepPCB images; some defects artificially augmented by dataset authors", "No factory deployment or live-camera validation", "Image-disjoint protocol; board groups can overlap", "CPU portfolio subset; see data audit for exact counts"]}
    (ROOT / "models/model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
