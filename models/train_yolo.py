"""Train a YOLOv8 detector (Ultralytics) and log its test-set scores.

Plain English: start from a model already trained on everyday photos
(yolov8n.pt), then keep training it on our steel images so it learns defects.

Usage:
    python -m models.train_yolo --config configs/yolov8n_merged_v1.yaml --device 0
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from models.utils import (load_config, log_experiment, resolved_data_yaml,
                          save_json, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolov8n_merged_v1.yaml")
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data = resolved_data_yaml(cfg["data"])
    model = YOLO(cfg["model"])
    model.train(
        data=data, epochs=cfg["epochs"], imgsz=cfg["imgsz"],
        batch=cfg["batch"], patience=cfg.get("patience", 50),
        workers=cfg.get("workers", 2), seed=cfg["seed"], deterministic=True,
        # absolute path: newer Ultralytics adds a "detect/" folder to relative ones
        project=str(Path(cfg["project"]).resolve()), name=cfg["run_name"],
        exist_ok=True, device=args.device,
    )

    # Ask the trainer where it really saved the weights, then score on test.
    best_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("Scoring", best_path)
    best = YOLO(str(best_path))
    m = best.val(data=data, split=cfg["eval_split"],
                 imgsz=cfg["imgsz"], device=args.device)
    p, r = float(m.box.mp), float(m.box.mr)
    metrics = {
        "split": cfg["eval_split"], "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "seed": cfg["seed"],
    }
    print(metrics)
    save_json(metrics, f"results/{cfg['run_name']}.json")
    log_experiment(cfg["run_name"], args.config, metrics)


if __name__ == "__main__":
    main()
