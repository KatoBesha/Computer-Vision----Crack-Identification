# Low-Cost Crack Detection on Metal Surfaces

Folders: `data/` (Millie) · `models/` (Mwewa) · `app/` (Maneli) · `paper/` (everyone) · `notebooks/` (Colab runs)

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Runs (on data/merged/)
    python -m models.baseline_canny --config configs/baseline_canny.yaml
    python -m models.train_yolo --config configs/yolov8n_merged_v1.yaml --device 0
    python -m models.eval_yolo --weights <path to best.pt> --run-name yolov8n_merged_v1 --split test --device 0

Every run appends a row to `results/experiment_log.csv` (config, git commit, scores).
Change settings by copying a YAML in `configs/`, never by editing the code.

## Results so far (test split, 530 images, 61 crack-positive, 123 boxes)
| Run | Precision | Recall | F1 | mAP50 |
|---|---|---|---|---|
| baseline1_canny_merged | 0.006 | 0.057 | 0.010 | |
| yolov8n_merged_v1 (50 epochs) | 0.251 | 0.463 | 0.326 | 0.259 |

Splits are not frozen yet, so these numbers are provisional.
