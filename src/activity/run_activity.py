"""
Run Activity

Purpose: Run ActivityRecognizer over test images and save results.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Run ActivityRecognizer over test images and save results.

Uses `ActivityRecognizer` to process every image in
`datasets/test/images` and writes CSV/JSON summaries to
`outputs/activity/`.
"""

import os
import csv
import json

from src.activity.activity_recognizer import ActivityRecognizer

IMAGE_FOLDER = "datasets/test/images"

OUTPUT_FOLDER = "outputs/activity"

CSV_FILE = os.path.join(OUTPUT_FOLDER, "activity_results.csv")
JSON_FILE = os.path.join(OUTPUT_FOLDER, "activity_results.json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

recognizer = ActivityRecognizer("config.yaml")

results_csv = []
results_json = []

images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"\nFound {len(images)} images\n")

for i, image in enumerate(images, start=1):

    image_path = os.path.join(IMAGE_FOLDER, image)

    result = recognizer.recognize(image_path)

    activity = result["activity"]

    attributes = result["attributes"]

    raw_scores = result["raw_scores"]

    smoothed_scores = result["smoothed_scores"]

    confidence = 0

    if len(smoothed_scores) > 0:
        confidence = max(smoothed_scores.values())

    print(f"[{i}/{len(images)}] {image}")
    print(f"Activity : {activity}")
    print(f"Confidence : {confidence:.3f}")
    print("-" * 40)

    results_csv.append({
        "image": image,
        "activity": activity,
        "confidence": round(confidence,3)
    })

    results_json.append({
        "image": image,
        "activity": activity,
        "confidence": round(confidence,3),
        "attributes": attributes,
        "raw_scores": raw_scores,
        "smoothed_scores": smoothed_scores
    })

with open(CSV_FILE,"w",newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "activity",
            "confidence"
        ]
    )

    writer.writeheader()

    writer.writerows(results_csv)

with open(JSON_FILE,"w") as f:

    json.dump(results_json,f,indent=4)

print("\n============================")
print("Activity Recognition Completed")
print("============================")
print("CSV :",CSV_FILE)
print("JSON:",JSON_FILE)