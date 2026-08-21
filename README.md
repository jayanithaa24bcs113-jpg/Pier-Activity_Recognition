# 🏗️ Pier Monitoring System

An end-to-end computer vision and rule-based AI framework for **automated monitoring of bridge pier construction activities**. The system uses fine-tuned YOLOv8 object detection models combined with a rule engine and Bayesian temporal filtering to recognize construction activities from images and video feeds in real time.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Construction Stages](#supported-construction-stages)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [1. Single Image Inference (CLI)](#1-single-image-inference-cli)
  - [2. Video Processing (CLI)](#2-video-processing-cli)
  - [3. Web Application (Flask)](#3-web-application-flask)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Running Tests](#6-running-tests)
- [Configuration](#configuration)
- [Authors](#authors)

---

## Overview

The Pier Monitoring System automates the recognition of construction activities on bridge pier sites. Given an image or video frame, the system:

1. **Detects** objects (cranes, rebars, formwork, workers, concrete equipment, etc.) using YOLOv8.
2. **Extracts** spatial and geometric attributes from the detected bounding boxes.
3. **Evaluates** domain-specific construction rules defined in YAML files.
4. **Smooths** predictions temporally using a Bayesian probability filter (for video streams).
5. **Decides** the final activity label (e.g., *Pier Stem Casting*, *Pier Cap Reinforcement*, or *Idle*).

---

## Supported Construction Stages

| Stage | Activity                    | Classes Detected                                                     |
| :---: | :-------------------------- | :------------------------------------------------------------------- |
|   1   | Pier Stem Reinforcement     | Crane, Pile Cap, Pier Rebar, Vertical Rebars, Worker                 |
|   2   | Pier Stem Casting           | Casted Pier, Concrete Pump, Formwork, Fresh Concrete, Needle Vibrator, Transit Mixer, Worker |
|   3   | Pier Cap Reinforcement      | Crane, Horizontal Rebar, Pier Stem, Worker, Rebar Cage               |
|   4   | Pier Cap Casting            | Casted Cap, Concrete Pump, Cap Formwork, Needle Vibrator, Pier Stem, Transit Mixer, Worker |

---

## Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Input Image │───▶│  YOLOv8 Detector  │───▶│  Attribute   │───▶│   Rule Engine    │───▶│   Bayesian   │
│  or Video    │    │  (per stage)      │    │  Extractor   │    │  (YAML rules)    │    │   Filter     │
└──────────────┘    └───────────────────┘    └──────────────┘    └──────────────────┘    └──────┬───────┘
                                                                                               │
                                                                                               ▼
                                                                                      ┌──────────────────┐
                                                                                      │  Activity Label  │
                                                                                      │  & Visualization │
                                                                                      └──────────────────┘
```

---

## Project Structure

```
Pier_Monitoring/
│
├── config.yaml                            # Primary system configuration (models, classes, thresholds)
├── requirements.txt                       # Python dependencies
├── main.py                                # CLI — single-image inference
├── process_video.py                       # CLI — video processing
├── evaluate.py                            # CLI — model/pipeline evaluation
├── train.py                               # CLI — YOLOv8 model training
├── debug_stage3.py                        # Debug helper for Stage 3
├── yolov8m.pt                             # Base YOLOv8-Medium pretrained weights
├── yolov8s.pt                             # Base YOLOv8-Small pretrained weights
│
├── configs/                               # YOLO dataset configuration files
│   ├── dataset.yaml                       #   Stage 1 — Pier Stem Reinforcement
│   ├── casting_dataset.yaml               #   Stage 2 — Pier Stem Casting
│   ├── cap_reinforcement_dataset.yaml     #   Stage 3 — Pier Cap Reinforcement
│   └── cap_casting_dataset.yaml           #   Stage 4 — Pier Cap Casting
│
├── src/                                   # Core source code package
│   ├── activity/                          #   Activity recognition orchestrators
│   │   ├── activity_recognizer.py         #     Base + stage-specific recognizer classes
│   │   ├── extract_attributes.py          #     Attribute extraction helpers
│   │   └── run_activity.py                #     Standalone activity runner
│   │
│   ├── attributes/                        #   Feature & attribute extraction
│   │   ├── attribute_extractor.py         #     Spatial/entity feature calculator
│   │   └── utils.py                       #     Geometry and bounding-box utilities
│   │
│   ├── bayesian/                          #   Temporal Bayesian smoothing
│   │   ├── bayesian_filter.py             #     Bayesian probability update filter
│   │   └── run_bayesian.py                #     Bayesian test runner
│   │
│   ├── detection/                         #   Object detection wrapper
│   │   ├── detector.py                    #     Ultralytics YOLO inference wrapper
│   │   └── utils.py                       #     Box parsing and NMS helpers
│   │
│   ├── evaluation/                        #   Benchmarking & metrics
│   │   ├── evaluator.py                   #     DatasetEvaluator orchestrator
│   │   ├── dataset_evaluator.py           #     Benchmark dataset processing loop
│   │   ├── metrics.py                     #     Precision, Recall, F1, mAP
│   │   └── report_generator.py            #     Evaluation summary & CSV reports
│   │
│   ├── inference/                         #   Bulk inference
│   │   └── inference.py                   #     Batch inference pipeline
│   │
│   ├── rules/                             #   Rule engine & rule definitions
│   │   ├── rule_engine.py                 #     Evaluates YAML rules against attributes
│   │   ├── run_rule_engine.py             #     Rule testing script
│   │   ├── rules.yaml                     #     Stage 1 rules (Stem Reinforcement)
│   │   ├── rules_stem_casting.yaml        #     Stage 2 rules (Stem Casting)
│   │   ├── rules_cap_reinforcement.yaml   #     Stage 3 rules (Cap Reinforcement)
│   │   └── rules_cap_casting.yaml         #     Stage 4 rules (Cap Casting)
│   │
│   ├── utils/                             #   Shared utilities
│   │   ├── config_loader.py               #     YAML configuration parser
│   │   ├── dataset_validator.py           #     Validates YOLO dataset folder structure
│   │   ├── file_utils.py                  #     File I/O helpers
│   │   ├── find_missing_file.py           #     Dataset integrity checker
│   │   ├── logger.py                      #     Logging setup
│   │   └── metrics.py                     #     Utility metric functions
│   │
│   ├── video/                             #   Video input/output processing
│   │   ├── video_processor.py             #     OpenCV video stream reader & annotator
│   │   ├── frame_processor.py             #     Per-frame processing pipeline
│   │   └── video_utils.py                 #     Video codec & frame rate helpers
│   │
│   └── visualization/                     #   Output rendering
│       └── visualizer.py                  #     Bounding box & label overlay drawer
│
├── tests/                                 # Automated test suite (pytest)
│   ├── test_detector.py                   #   YOLO detector tests
│   ├── test_rule_engine.py                #   Rule engine tests
│   ├── test_bayesian_filter.py            #   Bayesian filter tests
│   ├── test_video.py                      #   Video processing tests
│   ├── test_evaluation.py                 #   Evaluation metrics tests
│   ├── test_casting_activity.py           #   Stem Casting integration tests
│   ├── test_casting_attributes.py         #   Stem Casting attributes tests
│   ├── test_casting_rules.py              #   Stem Casting rules tests
│   ├── test_cap_reinforcement_activity.py #   Cap Reinforcement activity tests
│   ├── test_cap_reinforcement_attributes.py #  Cap Reinforcement attributes tests
│   ├── test_cap_reinforcement_rules.py    #   Cap Reinforcement rules tests
│   ├── test_cap_casting_activity.py       #   Cap Casting activity tests
│   ├── test_cap_casting_rules.py          #   Cap Casting rules tests
│   └── config_*.yaml                      #   Test configuration fixtures
│
├── webapp/                                # Flask web application
│   ├── app.py                             #   Flask backend (routes, inference endpoints)
│   ├── templates/
│   │   └── index.html                     #   Main web dashboard template
│   └── static/                            #   Uploaded images/videos & inference results
│
├── datasets/                              # Training & validation image datasets
├── outputs/                               # Evaluation results & CSV metric logs
├── runs/                                  # Trained YOLO model checkpoints (best.pt)
├── models/                                # Saved model directory
└── docs/
    └── documentation.md                   # Full API documentation
```

---

## Prerequisites

- **Python 3.10+** (this project uses Python 3.13)
- **pip** (Python package manager)
- **GPU (optional)** — CUDA-compatible GPU recommended for training; CPU works for inference

> **Note (Windows):** On this system, use `python` (not `python3`) to run commands.

---

## Installation

### 1. Clone or navigate to the project directory

```bash
cd Pier_Monitoring
```

### 2. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (CMD):
.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package         | Purpose                              |
| :-------------- | :----------------------------------- |
| `ultralytics`   | YOLOv8 object detection framework    |
| `opencv-python` | Image and video processing           |
| `pyyaml`        | YAML configuration file parsing      |
| `numpy`         | Numerical computation                |
| `pandas`        | Data analysis and CSV output         |
| `scikit-learn`  | Evaluation metrics                   |
| `flask`         | Web application framework            |

---

## How to Run

> **Important:** All commands below must be run from the **project root directory** (`Pier_Monitoring/`).

### 1. Single Image Inference (CLI)

Run the full detection → attribute extraction → rules → Bayesian → decision pipeline on a single image:

```bash
# Stage 1 — Pier Stem Reinforcement (default)
python main.py --source path/to/image.jpg

# Stage 2 — Pier Stem Casting
python main.py --activity casting --source path/to/image.jpg

# Stage 3 — Pier Cap Reinforcement
python main.py --activity cap_reinforcement --source path/to/image.jpg

# Stage 4 — Pier Cap Casting
python main.py --activity cap_casting --source path/to/image.jpg
```

**Options:**
| Flag           | Description                                              | Default          |
| :------------- | :------------------------------------------------------- | :--------------- |
| `--source`     | Path to the input image (**required**)                   | —                |
| `--activity`   | `reinforcement`, `casting`, `cap_reinforcement`, `cap_casting` | `reinforcement` |
| `--config`     | Path to configuration file                               | `config.yaml`    |

---

### 2. Video Processing (CLI)

Process an entire video frame-by-frame and produce an annotated output video:

```bash
# Stage 1 — Pier Stem Reinforcement (default)
python process_video.py --source path/to/video.mp4

# Stage 2 — Pier Stem Casting
python process_video.py --activity casting --source path/to/video.mp4

# Stage 3 — Pier Cap Reinforcement
python process_video.py --activity cap_reinforcement --source path/to/video.mp4

# Stage 4 — Pier Cap Casting
python process_video.py --activity cap_casting --source path/to/video.mp4

# Interactive mode (will prompt for video path)
python process_video.py
```

---

### 3. Web Application (Flask)

Launch the interactive web dashboard for image/video uploads and real-time activity detection:

```bash
# Navigate to the webapp directory and run
cd webapp
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

The web app features:
- 📷 **Image upload** — upload a construction site image to detect and classify activities
- 🎥 **Video upload** — process video files with frame-by-frame analysis
- 🔄 **Auto-detection** — automatically runs all trained activity models and picks the best match
- 📊 **Results dashboard** — annotated images, confidence scores, attribute details, and rule evaluations

---

### 4. Model Training

Train YOLOv8 models for each construction stage:

```bash
# Stage 1 — Pier Stem Reinforcement (default)
python train.py --activity reinforcement

# Stage 2 — Pier Stem Casting
python train.py --activity casting

# Stage 3 — Pier Cap Reinforcement
python train.py --activity cap_reinforcement

# Stage 4 — Pier Cap Casting
python train.py --activity cap_casting
```

Training parameters (epochs, batch size, image size, learning rate, patience, etc.) are configured in [config.yaml](config.yaml) under the corresponding `*_training` sections.

Trained model weights are saved to: `runs/detect/models/<activity_name>/weights/best.pt`

---

### 5. Model Evaluation

Evaluate a trained model on its test dataset:

```bash
# Stage 1 — Pier Stem Reinforcement (default)
python evaluate.py --activity reinforcement

# Stage 2 — Pier Stem Casting
python evaluate.py --activity casting

# Stage 3 — Pier Cap Reinforcement
python evaluate.py --activity cap_reinforcement

# Stage 4 — Pier Cap Casting
python evaluate.py --activity cap_casting
```

Evaluation results and metrics (Precision, Recall, F1, mAP) are saved to the `outputs/` directory.

---

### 6. Running Tests

Run the full automated test suite with pytest:

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_rule_engine.py -v
python -m pytest tests/test_bayesian_filter.py -v

# Run tests for a specific stage
python -m pytest tests/test_casting_activity.py -v
python -m pytest tests/test_cap_reinforcement_rules.py -v
python -m pytest tests/test_cap_casting_rules.py -v
```

---

## Configuration

All system settings are centralised in [config.yaml](config.yaml). Key sections:

| Section                        | Purpose                                                  |
| :----------------------------- | :------------------------------------------------------- |
| `model`                        | Default model weights path and save directory            |
| `dataset`                      | Dataset root and train/valid/test image paths            |
| `training`                     | Default training hyperparameters                         |
| `classes`                      | Class index → name mapping (must match YOLO label indices)|
| `activity`                     | Activity name, confidence threshold, temporal window     |
| `activities`                   | Multi-activity configuration (one block per stage)       |
| `activities.<stage>.rules`     | Path to the YAML rule file for that stage                |
| `activities.<stage>.model_trained` | Flag to enable/disable a stage (flip after training) |
| `casting_training`             | Training hyperparameters for Stage 2                     |
| `cap_reinforcement_training`   | Training hyperparameters for Stage 3                     |
| `cap_casting_training`         | Training hyperparameters for Stage 4                     |

---

## Authors

- **Jayanithaa** 
