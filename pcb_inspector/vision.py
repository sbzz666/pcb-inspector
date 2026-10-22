"""YOLO inference, overlap-tiled inference and template-difference baseline."""
from dataclasses import asdict, dataclass
import os
import threading
import time
import cv2
import numpy as np
from . import NAMES, ROOT

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "local_config"))


@dataclass
class Detection:
    box: list[float]
    class_id: int
    confidence: float
    source: str = "yolo"

    def to_dict(self):
        return {**asdict(self), "class_name": NAMES[self.class_id] if self.class_id >= 0 else "unclassified"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.0


def suppress(detections, threshold=0.45):
    """Class-aware global NMS after tile coordinates are mapped to original pixels."""
    kept = []
    for candidate in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(candidate.class_id != other.class_id or iou(candidate.box, other.box) <= threshold for other in kept):
            kept.append(candidate)
    return kept


def tile_starts(length, size, overlap):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Tile size must be positive, overlap must be in [0,1)")
    if length <= size:
        return [0]
    step = max(1, round(size*(1-overlap)))
    return sorted(set([*range(0, length-size+1, step), length-size]))


class Detector:
    def __init__(self, weights=None):
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(4)
        weights = weights or ROOT / "models/pcb_yolo11n.pt"
        if not os.path.isfile(weights):
            raise FileNotFoundError("PCB model missing. Run scripts/train.py first.")
        self.model = YOLO(str(weights))
        self.lock = threading.Lock()
        if list(self.model.names.values()) != NAMES:
            raise ValueError("Model classes do not match the PCB six-class schema")

    def predict(self, image, confidence=0.25, imgsz=320, tiled=False, tile_size=320, overlap=0.25):
        if image is None or image.ndim != 3 or image.size == 0:
            raise ValueError("Expected a non-empty BGR image")
        started = time.perf_counter()
        h, w = image.shape[:2]
        offsets = [(0, 0)] if not tiled else [(x, y) for y in tile_starts(h, tile_size, overlap) for x in tile_starts(w, tile_size, overlap)]
        detections = []
        with self.lock:
            for x, y in offsets:
                patch = image[y:y+tile_size, x:x+tile_size] if tiled else image
                result = self.model.predict(patch, conf=confidence, iou=0.45, imgsz=imgsz, device="cpu", verbose=False)[0]
                for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    mapped = [max(0, box[0]+x), max(0, box[1]+y), min(w, box[2]+x), min(h, box[3]+y)]
                    if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
                        detections.append(Detection(mapped, int(cls), float(score)))
        return suppress(detections), {"elapsed_ms": (time.perf_counter()-started)*1000, "tiles": len(offsets), "mode": "tiled" if tiled else "full", "imgsz": imgsz, "device": "cpu", "confidence": confidence}


def template_difference(image, template, min_area=8, tolerance=1, align=False):
    """Locate changes only: this baseline does NOT classify the six defect types.

    White-foreground differences tolerate a one-pixel edge displacement. No ground
    truth is used. Optional ECC registration rejects implausibly large translations.
    """
    started = time.perf_counter()
    if image.shape != template.shape:
        raise ValueError("待测图和模板尺寸必须一致")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    shift = [0.0, 0.0]
    if align:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(gray, reference, warp, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5))
        except cv2.error as exc:
            raise ValueError("模板配准失败，请检查图像是否来自同一电路板") from exc
        shift = warp[:, 2].tolist()
        if correlation < 0.5 or max(map(abs, shift)) > 20:
            raise ValueError("模板配准不可靠：相关性过低或位移超过 20 像素")
        reference = cv2.warpAffine(reference, warp, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    _, a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, b = cv2.threshold(reference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    added = cv2.bitwise_and(a, cv2.bitwise_not(cv2.dilate(b, kernel)))
    removed = cv2.bitwise_and(b, cv2.bitwise_not(cv2.dilate(a, kernel)))
    mask = cv2.bitwise_or(added, removed)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    h, w = gray.shape
    for x, y, width, height, area in stats[1:]:
        if area >= min_area:
            detections.append(Detection([max(0, int(x)-6), max(0, int(y)-6), min(w, int(x+width)+6), min(h, int(y+height)+6)], -1, 1.0, "opencv-difference"))
    return detections, mask, {"elapsed_ms": (time.perf_counter()-started)*1000, "registration_shift": shift, "min_area": min_area, "tolerance": tolerance, "classification": False, "confidence_note": "1.0 is a placeholder, not a calibrated probability"}


def draw(image, detections):
    canvas = image.copy()
    palette = [(87, 210, 61), (63, 99, 245), (230, 180, 38), (220, 106, 180), (30, 188, 250), (206, 133, 81)]
    for index, det in enumerate(detections, 1):
        color = palette[det.class_id] if det.class_id >= 0 else (40, 190, 240)
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{index} {NAMES[det.class_id]} {det.confidence:.2f}" if det.class_id >= 0 else f"{index} change"
        cv2.putText(canvas, label, (x1, max(12, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas
