"""Shared helpers: seeding, config loading, dataset paths, experiment log."""
import csv
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

LOG_PATH = Path("results/experiment_log.csv")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = 0) -> None:
    """Fix every random number generator we use so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not needed for the Canny baseline


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "no-git"


def list_images(data_yaml: str, split: str):
    """Return the image paths of one split of a YOLO-format dataset.

    Works with both layouts we will meet:
      1. Millie's merged set: data.yaml says `val: val.txt`, a text file that
         lists one image path per line ("./images/<source>/<id>.jpg").
      2. A Roboflow export: a folder of images per split (train/valid/test).
    'val' and 'valid' are treated as the same split name.
    """
    yaml_path = Path(data_yaml).resolve()
    root = yaml_path.parent
    cfg = load_config(str(yaml_path))
    names = [split] + {"val": ["valid"], "valid": ["val"]}.get(split, [])
    for key in names:
        entry = cfg.get(key)
        if isinstance(entry, str):
            target = root / entry
            if target.is_file() and target.suffix == ".txt":
                lines = [l.strip() for l in target.read_text().splitlines() if l.strip()]
                return [target.parent / l[2:] if l.startswith("./") else Path(l)
                        for l in lines]
            if target.is_dir():
                return sorted(p for p in target.iterdir() if p.suffix.lower() in IMG_EXT)
    for key in names:  # Roboflow fallback: <root>/<split>/images
        d = root / key / "images"
        if d.is_dir():
            return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    raise FileNotFoundError(f"No images found for split '{split}' via {data_yaml}")


def label_path(img_path: Path) -> Path:
    """.../images/<x>.jpg -> .../labels/<x>.txt (the rule Ultralytics uses)."""
    parts = list(Path(img_path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def resolved_data_yaml(data_yaml: str, out: str = "results/_resolved_data.yaml") -> str:
    """Copy data.yaml with `path:` made absolute.

    Millie's yaml says `path: .`. Some Ultralytics versions resolve a relative
    path against their own datasets folder instead of the yaml's folder, which
    makes train.txt "not found". An absolute path avoids that.
    """
    p = Path(data_yaml).resolve()
    cfg = load_config(str(p))
    if "path" in cfg:
        cfg["path"] = str(p.parent)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def read_yolo_labels(label_file: Path, img_w: int, img_h: int):
    """Read a YOLO label file -> list of (class_id, x1, y1, x2, y2) in pixels."""
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append((c,
                      (xc - w / 2) * img_w, (yc - h / 2) * img_h,
                      (xc + w / 2) * img_w, (yc + h / 2) * img_h))
    return boxes


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def log_experiment(run_name: str, config_path: str, metrics: dict) -> None:
    """Append one row to results/experiment_log.csv.

    Team rule: a result only counts if it is here with its config and commit.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": run_name,
        "git_commit": git_commit(),
        "config": config_path,
        **{k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in metrics.items()},
    }
    rows = []
    if LOG_PATH.exists():
        with open(LOG_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))  # union of columns
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
