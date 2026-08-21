"""
Extract Attributes

Purpose: Script to extract attributes from dataset using the trained YOLO model.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Script to extract attributes from dataset using the trained YOLO model.

This is a convenience script used during dataset processing. It loads a
trained model, runs inference on `datasets/test/images`, and writes
attributes to `outputs/attributes/` as CSV and JSON.
"""

import os
import csv
import json
from ultralytics import YOLO

from src.attributes.attribute_extractor import AttributeExtractor

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "runs/detect/models/pier_monitoring/weights/best.pt"

IMAGE_FOLDER = "datasets/test/images"

OUTPUT_FOLDER = "outputs/attributes"

CSV_FILE = os.path.join(OUTPUT_FOLDER, "attributes.csv")
JSON_FILE = os.path.join(OUTPUT_FOLDER, "attributes.json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading trained model...")

model = YOLO(MODEL_PATH)

extractor = AttributeExtractor(
    class_names=model.names,
    image_width=640,
    image_height=640
)

all_attributes = []

# --------------------------------------------------
# Process every image
# --------------------------------------------------

image_extensions = (".jpg", ".jpeg", ".png")

images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(image_extensions)
])

print(f"\nFound {len(images)} images\n")

for i, image_name in enumerate(images, start=1):

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    print(f"[{i}/{len(images)}] {image_name}")

    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False,
        device="cpu",
    )

    attributes = extractor.extract(results)

    attributes["image"] = image_name

    all_attributes.append(attributes)

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

if len(all_attributes) > 0:

    fieldnames = ["image"] + [
        key for key in all_attributes[0].keys()
        if key != "image"
    ]

    with open(CSV_FILE, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(all_attributes)

# --------------------------------------------------
# Save JSON
# --------------------------------------------------

with open(JSON_FILE, "w") as f:

    json.dump(all_attributes, f, indent=4)

print("\n===================================")
print("Attribute Extraction Completed")
print("===================================")
print(f"Images processed : {len(images)}")
print(f"CSV saved        : {CSV_FILE}")
print(f"JSON saved       : {JSON_FILE}")