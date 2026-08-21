"""
Find Missing File

Purpose: Simple script to list images and labels that don't match in training set.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Simple script to list images and labels that don't match in training set.

Use as a small utility during dataset preparation. This file is primarily a
script; import and refactor into functions if integrating into CI.
"""

from pathlib import Path

images = Path("datasets/train/images")
labels = Path("datasets/train/labels")

image_names = {
    f.stem
    for f in images.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
}

label_names = {
    f.stem
    for f in labels.iterdir()
    if f.suffix.lower() == ".txt"
}

print(f"Images : {len(image_names)}")
print(f"Labels : {len(label_names)}")

print("\nImages without labels:")
print(sorted(image_names - label_names))

print("\nLabels without images:")
print(sorted(label_names - image_names))
