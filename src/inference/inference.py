"""
Inference

Purpose: Inference runner for stored test dataset.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Inference runner for stored test dataset.

Runs YOLOv8 inference over images in the configured test folder and
writes a CSV file of detections to the activity-specific output directory.

Usage
-----
    python src/inference/inference.py --activity reinforcement
    python src/inference/inference.py --activity casting
    python src/inference/inference.py --activity cap_reinforcement
    python src/inference/inference.py --activity cap_casting

When ``--activity`` is omitted it defaults to ``reinforcement`` for full
backward compatibility with existing workflows.

Placeholder-safe: for ``cap_reinforcement`` or ``cap_casting`` (Stages 3
and 4), if the weights file does not exist on disk yet (model not
trained) or the test dataset folder is empty/missing, the script prints
a clear message and exits cleanly instead of raising an unhandled
exception.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

# =====================================================
# Argument Parsing
# =====================================================

parser = argparse.ArgumentParser(
    description="Run inference over a test dataset."
)

parser.add_argument(
    "--activity",
    type=str,
    default="reinforcement",
    choices=["reinforcement", "casting", "cap_reinforcement", "cap_casting"],
    help="Activity to run inference for (default: reinforcement).",
)

parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to config.yaml (default: config.yaml).",
)

args = parser.parse_args()

# =====================================================
# Load Configuration
# =====================================================

with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

activity_key = args.activity

# =====================================================
# Resolve Paths from Config
# =====================================================

# Activity-specific block — weights, output dir
activities_cfg = config.get("activities", {})
activity_cfg = activities_cfg.get(activity_key, {})

MODEL_PATH = activity_cfg.get(
    "weights",
    config["model"]["weights"]          # legacy fallback
)

OUTPUT_DIR = activity_cfg.get(
    "output_dir",
    "outputs"                           # legacy fallback
)

# Test images — read from dataset block keyed by activity.
# For reinforcement the existing dataset.test_path is used.
# For casting the casting_training.data yaml is read to find the test path.
# For cap_reinforcement the cap_reinforcement_training.data yaml is read.
# For cap_casting the cap_casting_training.data yaml is read.

if activity_key == "reinforcement":
    INPUT_FOLDER = Path(config["dataset"]["test_path"]) / "images"
    TRAINING_IMGSZ = config["training"]["imgsz"]

elif activity_key == "casting":
    # Read the casting dataset yaml to get the test path
    casting_data_yaml = config.get("casting_training", {}).get(
        "data", "configs/casting_dataset.yaml"
    )
    with open(casting_data_yaml, "r", encoding="utf-8") as f:
        casting_dataset = yaml.safe_load(f)

    dataset_root = casting_dataset.get("path", "datasets/Pier_stem_casting")
    test_rel = casting_dataset.get("test", "images/test")
    INPUT_FOLDER = Path(dataset_root) / test_rel
    TRAINING_IMGSZ = config.get("casting_training", {}).get("imgsz", 640)

elif activity_key == "cap_reinforcement":
    # cap_reinforcement (Stage 3) — read the cap reinforcement dataset yaml
    cap_training_cfg = config.get("cap_reinforcement_training", {})
    cap_data_yaml = cap_training_cfg.get(
        "data", "configs/cap_reinforcement_dataset.yaml"
    )

    if os.path.exists(cap_data_yaml):
        with open(cap_data_yaml, "r", encoding="utf-8") as f:
            cap_dataset = yaml.safe_load(f) or {}
        dataset_root = cap_dataset.get("path", "datasets/Pier_cap_reinforcement")
        test_rel = cap_dataset.get("test", "images/test")
    else:
        dataset_root = "datasets/Pier_cap_reinforcement"
        test_rel = "images/test"

    INPUT_FOLDER = Path(dataset_root) / test_rel
    TRAINING_IMGSZ = cap_training_cfg.get("imgsz", 640)

else:
    # cap_casting (Stage 4) — read the cap casting dataset yaml
    cap_casting_training_cfg = config.get("cap_casting_training", {})
    cap_casting_data_yaml = cap_casting_training_cfg.get(
        "data", "configs/cap_casting_dataset.yaml"
    )

    if os.path.exists(cap_casting_data_yaml):
        with open(cap_casting_data_yaml, "r", encoding="utf-8") as f:
            cap_casting_dataset = yaml.safe_load(f) or {}
        dataset_root = cap_casting_dataset.get(
            "path", "datasets/Pier_cap_casting"
        )
        test_rel = cap_casting_dataset.get("test", "images/test")
    else:
        dataset_root = "datasets/Pier_cap_casting"
        test_rel = "images/test"

    INPUT_FOLDER = Path(dataset_root) / test_rel
    TRAINING_IMGSZ = cap_casting_training_cfg.get("imgsz", 640)

OUTPUT_FOLDER = Path(OUTPUT_DIR) / "inference"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_FOLDER / "detections.csv"

# =====================================================
# Placeholder-Safe Weights Check
# =====================================================
# If the weights file for this activity does not exist on disk (e.g.
# Stage 3/4 before training), exit gracefully instead of letting
# YOLO(MODEL_PATH) raise an unhandled exception.

if not os.path.exists(MODEL_PATH):
    print(f"\nActivity       : {activity_key}")
    print(f"[WARNING] Weights not found at {MODEL_PATH}.")
    print(
        f"Train the model first using: "
        f"python train.py --activity {activity_key}\n"
    )
    exit()

# =====================================================
# Load YOLO Model
# =====================================================

print(f"\nActivity       : {activity_key}")
print(f"Loading model  : {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print(f"\nModel Path     : {MODEL_PATH}")
print(f"Input Path     : {INPUT_FOLDER}")
print(f"Output Path    : {OUTPUT_FOLDER}")

# =====================================================
# Collect Image Files
# =====================================================

extensions = ["*.jpg", "*.jpeg", "*.png"]

image_files = []
if INPUT_FOLDER.exists():
    for ext in extensions:
        image_files.extend(INPUT_FOLDER.glob(ext))

image_files = sorted(image_files)

print(f"\nTotal Images Found : {len(image_files)}")

if len(image_files) == 0:
    print("No images found. Check INPUT_FOLDER path.")
    exit()

# =====================================================
# Store Detections
# =====================================================

all_detections = []

start = time.time()

# =====================================================
# Run Inference
# =====================================================

for idx, image_path in enumerate(image_files, start=1):

    print(f"\n[{idx}/{len(image_files)}] {image_path.name}")

    results = model.predict(
        source=str(image_path),
        imgsz=TRAINING_IMGSZ,
        conf=0.25,
        save=True,
        project=str(OUTPUT_FOLDER),
        name="predict",
        exist_ok=True,
        verbose=False,
        device="cpu",
    )

    result = results[0]
    boxes = result.boxes

    if boxes is None:
        continue

    for box in boxes:

        cls  = int(box.cls.item())
        conf = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width    = x2 - x1
        height   = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        all_detections.append({
            "activity":    activity_key,
            "image":       image_path.name,
            "class_id":    cls,
            "class_name":  model.names[cls],
            "confidence":  round(conf, 4),
            "xmin":        round(x1, 2),
            "ymin":        round(y1, 2),
            "xmax":        round(x2, 2),
            "ymax":        round(y2, 2),
            "width":       round(width, 2),
            "height":      round(height, 2),
            "center_x":   round(center_x, 2),
            "center_y":   round(center_y, 2),
        })

# =====================================================
# Save CSV
# =====================================================

df = pd.DataFrame(all_detections)
df.to_csv(CSV_FILE, index=False)

end = time.time()

print("\n===================================")
print(f"Inference Completed — {activity_key}")
print("===================================")
print(f"Images Processed : {len(image_files)}")
print(f"Objects Detected : {len(df)}")
print(f"CSV Saved        : {CSV_FILE}")
print(f"Total Time       : {round(end - start, 2)} seconds")