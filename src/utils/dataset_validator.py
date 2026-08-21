"""
Dataset Validator

Purpose: Dataset validation helpers.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Dataset validation helpers.

`validate_dataset(data_dir)` performs basic checks for the expected
`train/valid/test` structure and prints mismatches. It raises on
missing required folders.
"""

from pathlib import Path


def validate_dataset(data_dir):
    data_dir = Path(data_dir)

    for split in ["train", "valid", "test"]:

        images_path = data_dir / split / "images"
        labels_path = data_dir / split / "labels"

        if not images_path.exists():
            raise ValueError(f"{images_path} does not exist")

        if not labels_path.exists():
            raise ValueError(f"{labels_path} does not exist")

        image_files = sorted([
            f for f in images_path.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        label_files = sorted(labels_path.glob("*.txt"))

        print(f"\n{split.upper()}")
        print(f"Images : {len(image_files)}")
        print(f"Labels : {len(label_files)}")

        image_names = {f.stem for f in image_files}
        label_names = {f.stem for f in label_files}

        missing_labels = image_names - label_names
        missing_images = label_names - image_names

        if missing_labels:
            print("\nImages without labels:")
            print(missing_labels)

        if missing_images:
            print("\nLabels without images:")
            print(missing_images)

        assert image_names == label_names, "Dataset mismatch!"

    print("\nDataset validation successful!")


if __name__ == "__main__":
    validate_dataset("datasets")