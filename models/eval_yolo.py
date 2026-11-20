"""Score an already-trained YOLO checkpoint on a split and log it.

Use this when training finished but the final scoring step did not run.

Usage:
    python -m models.eval_yolo --weights path/to/best.pt --run-name yolov8n_merged_v1 --split test --device 0
"""
import argparse

from ultralytics import YOLO

from models.utils import log_experiment, resolved_data_yaml, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", default="data/merged/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    m = YOLO(args.weights).val(data=resolved_data_yaml(args.data), split=args.split,
                               imgsz=args.imgsz, device=args.device)
    p, r = float(m.box.mp), float(m.box.mr)
    metrics = {
        "split": args.split, "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
    }
    print(metrics)
    save_json(metrics, f"results/{args.run_name}_{args.split}.json")
    log_experiment(args.run_name, "eval_yolo", metrics)


if __name__ == "__main__":
    main()
