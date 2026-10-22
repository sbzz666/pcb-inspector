"""Score-ordered one-to-one matching; each GT box can be matched at most once."""
from .vision import iou


def match(detections, ground_truth, threshold=0.5, class_aware=True):
    matched, tp, fp, missed = set(), [], [], []
    for pred_index in sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True):
        det = detections[pred_index]
        candidates = [(iou(det.box, gt["box"]), j) for j, gt in enumerate(ground_truth) if j not in matched and (not class_aware or gt["class_id"] == det.class_id)]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= threshold and best[1] >= 0:
            matched.add(best[1])
            tp.append(pred_index)
        else:
            fp.append(pred_index)
    missed = [j for j in range(len(ground_truth)) if j not in matched]
    return {"tp": len(tp), "fp": len(fp), "fn": len(missed), "matched_predictions": tp, "false_predictions": fp, "missed_ground_truth": missed}


def summarize(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
