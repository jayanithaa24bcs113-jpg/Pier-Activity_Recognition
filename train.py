"""
Train

Purpose: Training entrypoint for YOLOv8.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Training entrypoint for YOLOv8.

Calls `model.train()` with options from `config.yaml`. Supports all
four activities:

    python train.py --activity reinforcement
    python train.py --activity casting
    python train.py --activity cap_reinforcement
    python train.py --activity cap_casting

When ``--activity`` is omitted it defaults to ``reinforcement`` for full
backward compatibility with existing workflows.

Placeholder-safe: for ``cap_reinforcement`` or ``cap_casting`` (Stages 3
and 4), if the dataset folder's ``train/images`` does not exist or is
empty (dataset not yet annotated/exported), the script prints a clear
message and exits gracefully instead of raising an unhandled exception
from ``model.train()``.
"""

import argparse
import os

from ultralytics import YOLO
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_dataset_populated(train_images_dir: str) -> bool:
    """Check whether a dataset's train/images folder exists and has files.

    Args:
        train_images_dir: Path to the train images directory to check.

    Returns:
        bool: True if the directory exists and contains at least one
            file, False otherwise.
    """
    if not os.path.isdir(train_images_dir):
        return False
    return len(os.listdir(train_images_dir)) > 0


def train_model(config_path: str = "config.yaml", activity: str = "reinforcement"):
    """Train a YOLOv8 model for the selected activity.

    Args:
        config_path: Path to the main YAML configuration file.
        activity: ``"reinforcement"``, ``"casting"``,
            ``"cap_reinforcement"``, or ``"cap_casting"``. Defaults to
            ``"reinforcement"`` for backward compatibility — preserves
            the exact original training behaviour when called with no
            arguments.

    Returns:
        Training results object returned by ``model.train()``, or
        ``None`` if training was skipped (e.g. dataset not populated).
    """

    config = load_config(config_path)

    if activity == "cap_casting":
        # ------------------------------------------------------------
        # Cap casting training path (Stage 4) — placeholder-safe
        # ------------------------------------------------------------
        cap_casting_cfg = config.get("cap_casting_training", {})

        if not cap_casting_cfg:
            raise KeyError(
                "config.yaml is missing the 'cap_casting_training' "
                "block. Add it before training the cap casting model."
            )

        dataset_root = os.path.join("datasets", "Pier_cap_casting")
        train_images_dir = os.path.join(dataset_root, "train", "images")

        if not _is_dataset_populated(train_images_dir):
            print(
                "[ERROR] Dataset for cap_casting not found at "
                f"{dataset_root}\\\n"
                "Please annotate your images and export the dataset to "
                "this folder before training.\n"
                "Annotation guide: docs\\annotation_guide_stage3.md"
            )
            logger.warning(
                f"Cap casting training aborted — no images found "
                f"at {train_images_dir}."
            )
            return None

        data_yaml   = cap_casting_cfg.get("data", "configs/cap_casting_dataset.yaml")
        epochs      = cap_casting_cfg.get("epochs", 150)
        imgsz       = cap_casting_cfg.get("imgsz", 640)
        batch       = cap_casting_cfg.get("batch", 16)
        base_model  = cap_casting_cfg.get("model", "yolov8m.pt")
        run_name    = cap_casting_cfg.get("name", "pier_cap_casting")
        project_dir = cap_casting_cfg.get("project", "runs/detect")
        patience    = cap_casting_cfg.get("patience", 30)

        logger.info("Loading base YOLO model for cap casting training...")
        model = YOLO(base_model)

        logger.info("Starting cap casting training...")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            patience=patience,
            # Augmentation — mirrors cap_reinforcement/casting training
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )

        logger.info("Cap casting training completed successfully.")
        logger.info(
            f"Best model saved at:\n"
            f"{project_dir}/{run_name}/weights/best.pt"
        )
        logger.info(
            "Remember to set activities.cap_casting.model_trained: "
            "true in config.yaml to enable this stage in the pipeline."
        )

        return results

    if activity == "cap_reinforcement":
        # ------------------------------------------------------------
        # Cap reinforcement training path (Stage 3) — placeholder-safe
        # ------------------------------------------------------------
        cap_cfg = config.get("cap_reinforcement_training", {})

        if not cap_cfg:
            raise KeyError(
                "config.yaml is missing the 'cap_reinforcement_training' "
                "block. Add it before training the cap reinforcement model."
            )

        dataset_root = os.path.join("datasets", "Pier_cap_reinforcement")
        train_images_dir = os.path.join(dataset_root, "train", "images")

        if not _is_dataset_populated(train_images_dir):
            print(
                "[ERROR] Dataset for cap_reinforcement not found at "
                f"{dataset_root}\\\n"
                "Please annotate your images and export the dataset to "
                "this folder before training.\n"
                "Annotation guide: docs\\annotation_guide_stage3.md"
            )
            logger.warning(
                f"Cap reinforcement training aborted — no images found "
                f"at {train_images_dir}."
            )
            return None

        data_yaml   = cap_cfg.get("data", "configs/cap_reinforcement_dataset.yaml")
        epochs      = cap_cfg.get("epochs", 100)
        imgsz       = cap_cfg.get("imgsz", 640)
        batch       = cap_cfg.get("batch", 16)
        base_model  = cap_cfg.get("model", "yolov8m.pt")
        run_name    = cap_cfg.get("name", "pier_cap_reinforcement")
        project_dir = cap_cfg.get("project", "runs/detect")
        patience    = cap_cfg.get("patience", 30)

        logger.info("Loading base YOLO model for cap reinforcement training...")
        model = YOLO(base_model)

        logger.info("Starting cap reinforcement training...")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            patience=patience,
            # Augmentation — mirrors casting training, adapted for
            # rebar/cage geometry which is orientation-sensitive
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )

        logger.info("Cap reinforcement training completed successfully.")
        logger.info(
            f"Best model saved at:\n"
            f"{project_dir}/{run_name}/weights/best.pt"
        )
        logger.info(
            "Remember to set activities.cap_reinforcement.model_trained: "
            "true in config.yaml to enable this stage in the pipeline."
        )

        return results

    if activity == "casting":
        # ------------------------------------------------------------
        # Casting training path — uses casting_training block
        # ------------------------------------------------------------
        casting_cfg = config.get("casting_training", {})

        if not casting_cfg:
            raise KeyError(
                "config.yaml is missing the 'casting_training' block. "
                "Add it before training the casting model."
            )

        data_yaml   = casting_cfg.get("data", "configs/casting_dataset.yaml")
        epochs      = casting_cfg.get("epochs", 100)
        imgsz       = casting_cfg.get("imgsz", 640)
        batch       = casting_cfg.get("batch", 8)
        base_model  = casting_cfg.get("model", "yolov8m.pt")
        run_name    = casting_cfg.get("name", "pier_casting")
        project_dir = casting_cfg.get("project", "runs/detect/models")

        logger.info("Loading base YOLO model for casting training...")
        model = YOLO(base_model)

        logger.info("Starting casting training...")

        # In train_model(), casting branch — replace the model.train() call
        results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        # Augmentation to help with imbalance
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,    # helps rare classes appear more often
    )

        logger.info("Casting training completed successfully.")
        logger.info(
            f"Best model saved at:\n"
            f"{project_dir}/{run_name}/weights/best.pt"
        )

        return results

    # ------------------------------------------------------------------
    # Reinforcement training path — original logic preserved exactly
    # ------------------------------------------------------------------

    logger.info("Loading YOLO model...")

    model = YOLO(config["model"]["weights"])

    logger.info("Starting training...")

    results = model.train(
        data="configs/dataset.yaml",
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["imgsz"],
        batch=config["training"]["batch"],
        device=config["training"]["device"],
        optimizer=config["training"]["optimizer"],
        lr0=config["training"]["learning_rate"],
        patience=config["training"]["patience"],
        project=config["model"]["save_dir"],
        name="pier_monitoring",
        exist_ok=True,
    )

    logger.info("Training completed successfully.")

    logger.info(
        f"Best model saved at:\n"
        f"{config['model']['save_dir']}/pier_monitoring/weights/best.pt"
    )

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for pier monitoring."
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="reinforcement",
        choices=["reinforcement", "casting", "cap_reinforcement", "cap_casting"],
        help="Activity to train (default: reinforcement).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )

    args = parser.parse_args()

    train_model(config_path=args.config, activity=args.activity)