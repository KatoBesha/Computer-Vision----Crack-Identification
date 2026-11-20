"""Baseline 1: classical edge detection (Canny) + thresholding, no learning.

Idea in plain English: defects show up as strong edges. We find edges, merge
nearby edges into blobs, and call every big-enough blob a detection.
It is deliberately simple: it shows how far you get with no training, so the
YOLO results mean something.

We score box-level detection (precision / recall / F1 at IoU 0.5). Crack-free
images have no ground-truth boxes, so every blob found there is a false alarm.

Usage:
    python -m models.baseline_canny --config configs/baseline_canny.yaml
"""
import argparse
import itertools

import cv2

from models.utils import (label_path, list_images, load_config, log_experiment,
                          read_yolo_labels, save_json, set_seed)


def detect_boxes(gray, blur_ksize, canny_low, canny_high, close_ksize, min_area):
    """Image -> list of (x1, y1, x2, y2) boxes from Canny edges."""
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # join nearby edges
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes = []
    for i in range(1, n):  # label 0 is the background
        x, y, w, h, _ = stats[i]
        if w * h >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr):
    """Greedy one-to-one matching of predictions to ground truth.
    Returns (true positives, false positives, false negatives)."""
    used, tp = set(), 0
    for p in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            used.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def load_split(data_yaml, split):
    """Load all (gray image, ground-truth boxes) pairs of a split."""
    items = []
    for p in list_images(data_yaml, split):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        gts = [bx[1:] for bx in read_yolo_labels(label_path(p), w, h)]
        items.append((gray, gts))  # gts == [] for crack-free images
    if not items:
        raise RuntimeError(f"No images found for split '{split}'")
    return items


def score(items, params, min_area, thr):
    tp = fp = fn = 0
    for gray, gts in items:
        preds = detect_boxes(gray, min_area=min_area, **params)
        a, b, c = match(preds, gts, thr)
        tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_canny.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    tune = load_split(cfg["data"], cfg["tune_split"])
    test = load_split(cfg["data"], cfg["eval_split"])
    keys = list(cfg["grid"])
    best, best_params = None, None
    for values in itertools.product(*cfg["grid"].values()):
        params = dict(zip(keys, values))
        if params["canny_low"] >= params["canny_high"]:
            continue
        s = score(tune, params, cfg["min_box_area"], cfg["iou_threshold"])
        if best is None or s["f1"] > best["f1"]:
            best, best_params = s, params
    print(f"Best on {cfg['tune_split']}: {best_params} -> F1 {best['f1']:.3f}")

    final = score(test, best_params, cfg["min_box_area"], cfg["iou_threshold"])
    print(f"On {cfg['eval_split']}: precision {final['precision']:.3f}  "
          f"recall {final['recall']:.3f}  F1 {final['f1']:.3f}")

    save_json({"params": best_params, "tune": best, "test": final},
              f"results/{cfg['run_name']}.json")
    log_experiment(cfg["run_name"], args.config, {
        "split": cfg["eval_split"], "precision": final["precision"],
        "recall": final["recall"], "f1": final["f1"], "seed": cfg["seed"]})


if __name__ == "__main__":
    main()
