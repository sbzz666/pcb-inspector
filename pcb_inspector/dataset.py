"""Strict DeepPCB conversion with disjoint image IDs and exact-image hash audit."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from . import NAMES, ROOT


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Invalid image: {path}")
    return image


def read_labels(path, width, height):
    boxes = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.replace(",", " ").split()
        if len(values) != 5:
            raise ValueError(f"{path}:{number}: expected x1 y1 x2 y2 class")
        x1, y1, x2, y2, cls = map(int, values)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and 1 <= cls <= 6):
            raise ValueError(f"{path}:{number}: invalid box or class {values}")
        boxes.append({"box": [x1, y1, x2, y2], "class_id": cls - 1})
    return boxes


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box["box"]
    return f'{box["class_id"]} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}'


def prepare():
    raw = ROOT / "data/raw/DeepPCB"
    selection = json.loads((raw / "selection.json").read_text())
    trainval = selection["rows"]["trainval"]
    # Deterministic stratification by PCB group; original test list is never used in training.
    by_group = {}
    for row in trainval:
        by_group.setdefault(row.split("/")[0], []).append(row)
    splits = {"train": [], "val": [], "test": selection["rows"]["test"]}
    for group, rows in sorted(by_group.items()):
        rows = sorted(rows, key=lambda r: hashlib.sha256(("val42:" + r).encode()).hexdigest())
        n = max(1, round(len(rows) * 0.2)) if len(rows) > 1 else 0
        splits["val"].extend(rows[:n])
        splits["train"].extend(rows[n:])
    out = ROOT / "data/processed"
    # Refuse stale output that could silently reintroduce old train/test samples.
    if (out / "manifest.json").exists():
        raise FileExistsError("Prepared data exists. Use a new directory or explicitly remove data/processed before rebuilding.")
    records, ids, hashes = [], {}, {}
    for split, rows in splits.items():
        for folder in ["images", "labels", "templates"]:
            (out / folder / split).mkdir(parents=True, exist_ok=True)
        for row in sorted(rows):
            image_rel, label_rel = row.split()
            stub = raw / "PCBData" / Path(image_rel).with_suffix("")
            image_path = stub.parent / (stub.name + "_test.jpg")
            template_path = stub.parent / (stub.name + "_temp.jpg")
            ident = stub.name
            image = read_image(image_path)
            template = read_image(template_path)
            h, w = image.shape[:2]
            if image.shape != template.shape:
                raise ValueError(f"Template size mismatch: {ident}")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if ident in ids or digest in hashes:
                raise ValueError(f"Duplicate image ID or image pixels: {ident}")
            ids[ident], hashes[digest] = split, split
            boxes = read_labels(raw / "PCBData" / label_rel, w, h)
            target = out / "images" / split / f"{ident}.jpg"
            template_target = out / "templates" / split / f"{ident}.jpg"
            shutil.copy2(image_path, target)
            shutil.copy2(template_path, template_target)
            (out / "labels" / split / f"{ident}.txt").write_text("\n".join(yolo_line(b, w, h) for b in boxes) + "\n")
            records.append({"id": ident, "group": image_rel.split("/")[0], "split": split, "image": target.relative_to(ROOT).as_posix(), "template": template_target.relative_to(ROOT).as_posix(), "sha256_pixels": digest, "width": w, "height": h, "boxes": boxes})
    manifest = {"source": selection["source"], "full_dataset": selection["full"], "seed": 42, "protocol": "Original test membership; deterministic within-group 80/20 train/val. Image-disjoint, NOT board-group-disjoint.", "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "data.yaml").write_text(yaml.safe_dump({"path": out.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(NAMES))}, sort_keys=False), encoding="utf-8")
    audit = {"images": dict(Counter(r["split"] for r in records)), "boxes": {s: dict(Counter(NAMES[b["class_id"]] for r in records if r["split"] == s for b in r["boxes"])) for s in splits}, "duplicate_ids": 0, "duplicate_pixel_hashes": 0, "protocol": manifest["protocol"], "dataset_is_subset": not selection["full"], "archive_sha256": None}
    (ROOT / "reports/data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
