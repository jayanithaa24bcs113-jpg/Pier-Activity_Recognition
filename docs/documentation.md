# Pier Monitoring System - Codebase Documentation

This document provides a comprehensive overview of the purpose of every file, class, and function in the Pier Monitoring System, in compliance with the **Kumaraguru Institutions Project Development and Coding Guidelines**.

## Project Structure Overview

```
Project/
├── configs/            # Configuration files (dataset.yaml, casting_dataset.yaml)
├── docs/               # System documentation files
├── src/                # Core system source code
│   ├── activity/       # Activity recognition managers and pipelines
│   ├── attributes/     # Attribute extraction from bounding boxes
│   ├── bayesian/       # Bayesian temporal smoothing filters
│   ├── detection/      # YOLO model wrappers and inference helpers
│   ├── evaluation/     # Metrics calculation and validation reports
│   ├── inference/      # Bulk test dataset inference scripts
│   ├── rules/          # Rule engine for construction logic evaluation
│   ├── utils/          # Config loaders, logging, and dataset validators
│   └── video/          # Video frame extraction and processing pipeline
├── tests/              # Automated unit tests for all components
├── webapp/             # Flask web frontend for demonstations
├── config.yaml         # Main configuration settings
└── requirements.txt    # Project dependencies list
```

## Module Documentation Details

### Module: `evaluate.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Pier Reinforcement and casting|
| **Author** | jayanithaa|
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | CLI entry point to run dataset evaluation.<br>Evaluate<br>CLI entry point to run dataset evaluation.<br>Constructs a ``DatasetEvaluator`` for the selected activity and triggers<br>evaluation over the configured test dataset.<br>Usage<br>-----<br>python evaluate.py --activity casting<br>When ``--activity`` is omitted it defaults to ``reinforcement`` for full<br>backward compatibility. |

#### Functions in this file:

*   `main`: Parse CLI args and run dataset evaluation.

---

### Module: `main.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Main CLI entry point for single-image pier monitoring inference.<br>Main<br>Main CLI entry point for single-image pier monitoring inference.<br>Runs the full recognition pipeline (detect - extract - rules - bayesian<br>- decide) on a single image and prints the result.<br>Usage<br>-----<br>python main.py --source path/to/image.jpg<br>python main.py --activity casting --source path/to/image.jpg<br>When ``--activity`` is omitted it defaults to ``reinforcement`` for full<br>backward compatibility. |

#### Functions in this file:

*   `build_recognizer`: Return the correct recogniser for the selected activity.
*   `main`: Parse CLI args, run recognition on a single image, print results.

---

### Module: `process_video.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | CLI front-end to process a single input video.<br>Process Video<br>CLI front-end to process a single input video.<br>Runs full pier monitoring recognition over every frame of a video and<br>writes an annotated output video.<br>Usage<br>-----<br>python process_video.py --activity casting --source path/to/video.mp4<br>When ``--activity`` is omitted it defaults to ``reinforcement`` for full<br>backward compatibility. When ``--source`` is omitted the script falls<br>back to an interactive prompt, matching the original behaviour. |

#### Functions in this file:

*   `main`: Parse CLI args (or prompt), process the video, print summary.

---

### Module: `src/activity/activity_recognizer.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Activity Recognition Module for Pier Monitoring.<br>Activity Recognizer<br>Activity Recognition Module for Pier Monitoring.<br>Orchestrates detection, attribute extraction, rule evaluation, and<br>Bayesian smoothing to produce a final activity label for an image.<br>Two recognisers are provided:<br>``ActivityRecognizer``<br>Pier Stem Reinforcement (Stage 1).  Preserved exactly as before.<br>``CastingActivityRecognizer``<br>Pier Stem Casting (Stage 2).<br>Both inherit from ``_BaseActivityRecognizer`` which implements the shared<br>pipeline (detect - extract - rules - bayesian - decide).  Each subclass<br>supplies its own extractor, rule file, activity/idle labels, and decision<br>thresholds via ``_build_components`` and ``decide_activity``. |

#### Classes in this file:

##### `_BaseActivityRecognizer`
*   **Class Purpose:** Shared pipeline for all activity recognisers.<br><br>Subclasses must implement:<br>    - ``_build_components()`` — set ``self.detector``, ``self.extractor``,<br>      ``self.rule_engine``, ``self.activity_label``, ``self.idle_label``<br>    - ``decide_activity(attributes, scores)`` — map scores to a label<br><br>Args:<br>    config_path: Path to the main YAML configuration file.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `_build_components`: Initialise activity-specific components.
        *   `recognize`: Run the full recognition pipeline on a single image.
        *   `decide_activity`: Map smoothed scores and attributes to an activity label.
        *   `_mean_confidence`: Return mean detection confidence across all boxes.

##### `ActivityRecognizer`
*   **Class Purpose:** High-level pipeline coordinator for Pier Stem Reinforcement.<br><br>Methods<br>-------<br>recognize(image_path)<br>    Run full pipeline and return activity, attributes and scores.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `_build_components`: Initialise reinforcement-specific components.
        *   `decide_activity`: Decide reinforcement activity from smoothed scores.

##### `CastingActivityRecognizer`
*   **Class Purpose:** High-level pipeline coordinator for Pier Stem Casting.<br><br>Loads casting-specific weights, attribute extractor, and rules<br>from the ``activities.casting`` block in config.yaml.<br><br>Methods<br>-------<br>recognize(image_path)<br>    Run full pipeline and return activity, attributes and scores.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `_build_components`: Initialise casting-specific components.
        *   `decide_activity`: Decide casting activity from smoothed scores and attributes.

---

### Module: `src/activity/extract_attributes.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Script to extract attributes from dataset using the trained YOLO model.<br>Extract Attributes<br>Script to extract attributes from dataset using the trained YOLO model.<br>This is a convenience script used during dataset processing. It loads a<br>trained model, runs inference on `datasets/test/images`, and writes<br>attributes to `outputs/attributes/` as CSV and JSON. |

---

### Module: `src/activity/run_activity.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Run ActivityRecognizer over test images and save results.<br>Run Activity<br>Run ActivityRecognizer over test images and save results.<br>Uses `ActivityRecognizer` to process every image in<br>`datasets/test/images` and writes CSV/JSON summaries to<br>`outputs/activity/`. |

---

### Module: `src/attributes/attribute_extractor.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Attribute extractor for Pier Monitoring activities.<br>Attribute Extractor<br>Attribute extractor for Pier Monitoring activities.<br>Extracts high-level attributes from YOLO detections for activity recognition.<br>Two extractors are provided:<br>``AttributeExtractor``<br>Pier Stem Reinforcement (Stage 1).<br>Dataset classes: Crane, Pile cap, pier rebar, vertical rebars, worker.<br>``CastingAttributeExtractor``<br>Pier Stem Casting (Stage 2).<br>Dataset classes: Formwork, Concrete Pump, Transit Mixer, Vibrator,<br>Fresh Concrete, Worker, Casted Pier.<br>Both expose an ``extract(results)`` method that returns a dictionary of<br>named attributes consumed by the rule engine. |

#### Classes in this file:

##### `AttributeExtractor`
*   **Class Purpose:** Extract numerical attributes from YOLO detection results.<br><br>Parameters<br>----------<br>class_names: dict<br>    Mapping from class id to class name (model.names).<br>image_width, image_height: int<br>    Expected image dimensions for normalising area-based attributes.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `extract`: No docstring provided.

##### `CastingAttributeExtractor`
*   **Class Purpose:** Extract numerical attributes from YOLO detections for Pier Stem Casting.<br><br>Mirrors the interface of ``AttributeExtractor`` exactly: construct with<br>``class_names`` and optional image dimensions, then call<br>``extract(results)`` to receive a flat attribute dictionary.<br><br>Parameters<br>----------<br>class_names: dict<br>    Mapping from class id to class name (model.names).<br>image_width, image_height: int<br>    Image dimensions used to normalise area-based attributes.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `extract`: Extract casting-specific attributes from YOLO results.

#### Functions in this file:

*   `box_area`: Return the pixel area of a bounding box.
*   `centroid_distance`: Return the Euclidean distance between the centroids of two boxes.
*   `any_worker_near`: Return True if any worker centroid is within threshold of any target box.

---

### Module: `src/attributes/utils.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Attribute module utility helpers.<br>Utils<br>Attribute module utility helpers.<br>Contains helper functions for normalising and merging attribute<br>dictionaries used by the attribute extractor and rule engine. |

#### Functions in this file:

*   `normalise_score`: Clamp and normalise a score to [0, 1].
*   `merge_attributes`: Merge multiple attribute dictionaries by summing numeric values.

---

### Module: `src/bayesian/bayesian_filter.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Bayesian exponential smoothing filter for activity scores.<br>Bayesian Filter<br>Bayesian exponential smoothing filter for activity scores.<br>Maintains a running smoothed estimate of each named score using an<br>exponential moving average (EMA), which acts as a Bayesian update<br>with a uniform prior.<br>Module header added to satisfy coding standards. Class docstring<br>describes purpose; methods contain inline docstrings where needed. |

#### Classes in this file:

##### `BayesianFilter`
*   **Class Purpose:** Exponential-smoothing filter for per-activity confidence scores.<br><br>Each call to :meth:`update` blends the new raw scores with the existing<br>smoothed estimates::<br><br>    smoothed[k] = alpha * new[k] + (1 - alpha) * smoothed[k]<br><br>Args:<br>    alpha: Smoothing factor in (0, 1].  Higher values give more weight<br>           to recent observations; 1.0 disables smoothing entirely.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `update`: Incorporate new raw scores and return the updated smoothed scores.
        *   `reset`: Clear all smoothed scores (start fresh).

---

### Module: `src/bayesian/run_bayesian.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Run Bayesian smoothing over rule engine outputs.<br>Run Bayesian<br>Run Bayesian smoothing over rule engine outputs.<br>This script reads `outputs/rules/rule_results.csv`, applies the<br>`BayesianFilter`, and writes CSV/JSON outputs to<br>`outputs/bayesian/`. The top-level script is intentionally simple and<br>meant for command-line runs. Add CLI argument parsing if needed. |

---

### Module: `src/detection/detector.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Object detection module for Pier Monitoring.<br>Detector<br>Object detection module for Pier Monitoring.<br>Wraps a YOLOv8 model loaded from config.yaml.  Supports multi-activity<br>mode via the ``activity`` parameter, which selects the correct weights<br>and class names from the ``activities`` block in config.yaml.  When no<br>activity is supplied the legacy ``model.weights`` path is used so all<br>existing code continues to work without modification. |

#### Classes in this file:

##### `Detector`
*   **Class Purpose:** YOLOv8-based object detector.<br><br>Args:<br>    config_path: Path to the main YAML configuration file.<br>    activity: Activity key defined under ``activities`` in config.yaml<br>        (e.g. ``"reinforcement"`` or ``"casting"``).  When ``None``<br>        the legacy ``model.weights`` entry is used.<br><br>Attributes:<br>    classes: Dict mapping class index to class name string.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `_load_model`: Load YOLOv8 model weights from config.
        *   `_load_classes`: Load class index - name mapping from config.
        *   `detect`: Run inference on a single image.
        *   `verify_dataset`: Verify that a dataset directory has the expected structure.

---

### Module: `src/detection/utils.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Detection utility helpers for pier monitoring.<br>Utils<br>Detection utility helpers for pier monitoring.<br>Provides bounding-box conversion, NMS, and confidence-filtering helpers<br>that complement the main Detector class.<br>Module header added to comply with coding standards. Add a concise<br>description of responsibilities and public API if desired. |

#### Functions in this file:

*   `xyxy_to_xywh`: Convert [x1, y1, x2, y2] to [cx, cy, w, h].
*   `filter_by_confidence`: Filter detection results to keep only boxes above a confidence threshold.
*   `count_objects_by_class`: Count detected objects grouped by class name.
*   `main`: No docstring provided.

---

### Module: `src/evaluation/dataset_evaluator.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Dataset-level evaluation for pier monitoring activities.<br>Dataset Evaluator<br>Dataset-level evaluation for pier monitoring activities.<br>``DatasetEvaluator`` iterates over a labelled test dataset, runs the<br>full recognition pipeline on every image, and computes classification<br>metrics via ``ActivityEvaluator``.<br>Expected dataset structure<br>--------------------------<br>datasets/<activity>/test/<br>images/<br><label_class>/<br>image1.jpg<br>image2.jpg<br>...<br>The subfolder name under ``images/`` is used as the ground-truth label.<br>If images are not organised into subfolders a ``ground_truth_label``<br>fallback can be supplied at construction time.<br>Supports multiple activities via the ``activity`` parameter.<br>Defaults to ``"reinforcement"`` for backward compatibility. |

#### Classes in this file:

##### `DatasetEvaluator`
*   **Class Purpose:** Run evaluation over a full test dataset.<br><br>Args:<br>    activity: Activity key — ``"reinforcement"`` or ``"casting"``.<br>        Defaults to ``"reinforcement"`` for backward compatibility.<br>    config_path: Path to the main YAML configuration file.<br>    ground_truth_label: Fallback ground-truth label used when images<br>        are not organised into labelled subfolders.  When ``None``<br>        the activity label from config.yaml is used as the fallback.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `_resolve_test_folder`: Resolve the test images folder for the selected activity.
        *   `_build_recognizer`: Instantiate the correct recogniser for the selected activity.
        *   `_collect_images`: Collect all test images with their ground-truth labels.
        *   `evaluate`: Run the full evaluation pipeline over the test dataset.

---

### Module: `src/evaluation/evaluator.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Activity evaluation utilities.<br>Evaluator<br>Activity evaluation utilities.<br>Provides ``ActivityEvaluator`` which wraps ``MetricsCalculator`` to<br>accumulate predictions and generate summary metrics.<br>Supports multiple activities via the ``activity`` parameter which routes<br>output to the correct directory defined in config.yaml.  Defaults to<br>``"reinforcement"`` for full backward compatibility. |

#### Classes in this file:

##### `ActivityEvaluator`
*   **Class Purpose:** Activity Evaluation Engine.<br><br>Use ``update(ground_truth, prediction)`` to add a single prediction.<br>Call ``evaluate()`` to compute and persist metrics.<br><br>Args:<br>    activity: Activity key — ``"reinforcement"`` or ``"casting"``.<br>        Defaults to ``"reinforcement"`` for backward compatibility.<br>    config_path: Path to the main YAML configuration file.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `update`: Store one prediction.
        *   `evaluate`: Calculate and display evaluation metrics.
        *   `save_csv`: Save evaluation metrics to CSV in the activity output folder.

---

### Module: `src/evaluation/metrics.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Evaluation metrics utilities.<br>Metrics<br>Evaluation metrics utilities.<br>``MetricsCalculator`` collects true/predicted labels and computes standard<br>classification metrics (accuracy, precision, recall, F1, confusion matrix).<br>Works for any activity — operates on generic string labels with no<br>knowledge of which activity produced them. |

#### Classes in this file:

##### `MetricsCalculator`
*   **Class Purpose:** Calculate and store evaluation results for activity recognition.<br><br>Methods<br>-------<br>add_result(ground_truth, prediction)<br>    Append one sample's ground-truth and predicted label.<br>calculate_metrics()<br>    Compute and return a metrics dictionary.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `add_result`: Store one prediction.
        *   `calculate_metrics`: Calculate evaluation metrics.

---

### Module: `src/evaluation/report_generator.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | No docstring provided.<br>Report Generator |

---

### Module: `src/inference/inference.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Inference runner for stored test dataset.<br>Inference<br>Inference runner for stored test dataset.<br>Runs YOLOv8 inference over images in the configured test folder and<br>writes a CSV file of detections to the activity-specific output directory.<br>Usage<br>-----<br>python src/inference/inference.py --activity reinforcement<br>python src/inference/inference.py --activity casting<br>When ``--activity`` is omitted it defaults to ``reinforcement`` for full<br>backward compatibility with existing workflows. |

---

### Module: `src/rules/rule_engine.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Rule Engine for pier construction activity scoring.<br>Rule Engine<br>Rule Engine for pier construction activity scoring.<br>Rules are defined in a YAML file and describe conditions based on attributes<br>extracted from detection results.  Each matching rule contributes a numeric<br>score that downstream components (e.g. BayesianFilter) can consume.<br>Supports multiple activities by accepting a ``rules_file`` path at<br>initialisation.  When no ``rules_file`` is supplied the path is read from<br>``config['rules']['file']`` preserving full backward compatibility.<br>The ``apply_rules`` method accepts an optional ``obj_conf`` value used to<br>weight the ``Tall Vertical Shuttering`` rule score by object confidence,<br>as specified in the Pier Stem Casting rule logic. |

#### Classes in this file:

##### `RuleEngine`
*   **Class Purpose:** Evaluate attribute dictionaries against configured rules.<br><br>Args:<br>    config_path: Path to the main YAML configuration file.<br>    rules_file: Optional direct path to a rules YAML file.  When<br>        supplied this takes precedence over ``config['rules']['file']``<br>        so each activity can load its own rules without changing config.<br><br>Methods<br>-------<br>apply_rules(attributes, obj_conf)<br>    Return mapping of rule name -> score for matched rules.

    *   **Methods:**
        *   `__init__`: Initialise the RuleEngine.
        *   `_load_rules`: Load rules from the YAML file at ``self.rules_file``.
        *   `apply_rules`: Evaluate all rules against the given attributes and return scores.

#### Functions in this file:

*   `_evaluate_condition`: Recursively evaluate a single condition dictionary.

---

### Module: `src/rules/run_rule_engine.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Run the rule engine against extracted attributes CSV.<br>Run Rule Engine<br>Run the rule engine against extracted attributes CSV.<br>Reads `outputs/attributes/attributes.csv`, evaluates configured rules,<br>and writes CSV/JSON summaries to `outputs/rules/`. |

---

### Module: `src/utils/config_loader.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Configuration loader utility.<br>Config Loader<br>Configuration loader utility.<br>Provide `load_config(config_path)` that reads YAML and returns a<br>configuration dictionary. Raises `FileNotFoundError` if the file is<br>missing to fail-fast during startup. |

#### Functions in this file:

*   `load_config`: Load YAML configuration file.

---

### Module: `src/utils/dataset_validator.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Dataset validation helpers.<br>Dataset Validator<br>Dataset validation helpers.<br>`validate_dataset(data_dir)` performs basic checks for the expected<br>`train/valid/test` structure and prints mismatches. It raises on<br>missing required folders. |

#### Functions in this file:

*   `validate_dataset`: No docstring provided.

---

### Module: `src/utils/file_utils.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | File utilities used across the project.<br>File Utils<br>File utilities used across the project.<br>Provides small helpers for saving metrics and ensuring directories exist. |

#### Functions in this file:

*   `save_metrics_to_csv`: Save a metrics DataFrame to a CSV file.
*   `ensure_dir`: Create a directory (and any parents) if it does not exist.

---

### Module: `src/utils/find_missing_file.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Simple script to list images and labels that don't match in training set.<br>Find Missing File<br>Simple script to list images and labels that don't match in training set.<br>Use as a small utility during dataset preparation. This file is primarily a<br>script; import and refactor into functions if integrating into CI. |

---

### Module: `src/utils/logger.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Logging utilities.<br>Logger<br>Logging utilities.<br>Provides `get_logger(name, level)` which returns a configured logger<br>with both console and daily-rotated file handlers (basic rotation by<br>date). Avoid creating duplicate handlers on repeated imports. |

#### Functions in this file:

*   `get_logger`: Create and return a logger with console and file handlers.

---

### Module: `src/utils/metrics.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Training metrics extraction helpers.<br>Metrics<br>Training metrics extraction helpers.<br>`calculate_metrics(results)` reads common YOLOv8 result fields and<br>returns a single-row DataFrame. Missing values are replaced with NaN<br>and a warning is logged. |

#### Functions in this file:

*   `calculate_metrics`: Calculate and return training metrics from YOLOv8 results.

---

### Module: `src/video/__inti__.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Video package initializer.<br>Inti<br>Video package initializer.<br>Note: filename appears to be misspelled (`__inti__.py`); consider<br>renaming to `__init__.py` to ensure the package imports correctly. |

---

### Module: `src/video/frame_processor.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Frame Processor used by video pipeline.<br>Frame Processor<br>Frame Processor used by video pipeline.<br>``FrameProcessor.process_frame(frame)`` saves the frame to a temporary<br>file, runs the image through the activity recognition pipeline, and<br>returns activity information and detections.<br>Supports multiple activities via the ``activity`` parameter which selects<br>the correct recogniser.  Defaults to ``reinforcement`` for full backward<br>compatibility. |

#### Classes in this file:

##### `FrameProcessor`
*   **Class Purpose:** Process individual video frames through the recognition pipeline.<br><br>Args:<br>    activity: Activity key — ``"reinforcement"`` or ``"casting"``.<br>        Defaults to ``"reinforcement"`` for backward compatibility.<br>    config_path: Path to the main YAML configuration file.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `_build_recognizer`: Instantiate the correct recogniser for the selected activity.
        *   `process_frame`: Process one video frame through the recognition pipeline.

---

### Module: `src/video/video_processor.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Video processing pipeline.<br>Video Processor<br>Video processing pipeline.<br>``VideoProcessor.process_video(input_video, output_video)`` processes an<br>input video frame-by-frame using ``FrameProcessor`` and writes an<br>annotated output video.  Returns a stats dictionary summarising the run.<br>Supports multiple activities via the ``activity`` parameter.  Defaults to<br>``"reinforcement"`` for full backward compatibility. |

#### Classes in this file:

##### `VideoProcessor`
*   **Class Purpose:** Process a video file through the pier monitoring pipeline.<br><br>Args:<br>    activity: Activity key — ``"reinforcement"`` or ``"casting"``.<br>        Defaults to ``"reinforcement"`` for backward compatibility.<br>    config_path: Path to the main YAML configuration file.

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `process_video`: Process a video file and write an annotated output video.

---

### Module: `src/video/video_utils.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Video utility helpers.<br>Video Utils<br>Video utility helpers.<br>Provide `open_video` and `create_video_writer` helpers used by the<br>video processing pipeline. |

#### Functions in this file:

*   `open_video`: Open a video.
*   `create_video_writer`: Create output video writer.

---

### Module: `src/visualization/visualizer.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Visualizer for pier monitoring detection results.<br>Visualizer<br>Visualizer for pier monitoring detection results.<br>Draws bounding boxes and activity labels on images and either displays<br>them in a window or saves them to disk.<br>Main class `Visualizer` exposes `draw`, `display`, and `save` methods. |

#### Classes in this file:

##### `Visualizer`
*   **Class Purpose:** Draw detection results and activity labels on images.<br><br>Args:<br>    config_path: Path to the main YAML configuration file.<br>                 Used to resolve class names. Pass ``None`` to skip<br>                 config loading (class names fall back to indices).

    *   **Methods:**
        *   `__init__`: No docstring provided.
        *   `draw`: Draw boxes and activity label on an image and return it.
        *   `display`: Draw results and show the annotated image in an OpenCV window.
        *   `save`: Draw results and save the annotated image to disk.

---

### Module: `train.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Training entrypoint for YOLOv8.<br>Train<br>Training entrypoint for YOLOv8.<br>Calls `model.train()` with options from `config.yaml`. Supports both<br>activities:<br>python train.py --activity reinforcement<br>python train.py --activity casting<br>When ``--activity`` is omitted it defaults to ``reinforcement`` for full<br>backward compatibility with existing workflows. |

#### Functions in this file:

*   `train_model`: Train a YOLOv8 model for the selected activity.

---

### Module: `webapp/app.py`
| Detail | Value |
| :--- | :--- |
| **Module Name** | Unknown |
| **Author** | Student & Antigravity |
| **Date** | 2026-07-02 |
| **Version** | 1.0.0 |
| **Purpose** | Flask webapp front-end for Pier Monitoring demo.<br>App<br>Flask webapp front-end for Pier Monitoring demo.<br>Handles image and video uploads, runs the recognition pipeline across<br>ALL configured activities, and automatically determines which activity<br>is occurring — the user does not select an activity.<br>Supports any number of activities defined under ``activities`` in<br>config.yaml (currently Pier Stem Reinforcement and Pier Stem Casting;<br>designed to scale to additional stages without code changes). |

#### Functions in this file:

*   `_configured_activities`: Return the list of activity keys defined in config.yaml.
*   `_build_recognizer`: Instantiate the recogniser for a given activity key.
*   `_build_detector`: Return a YOLO model loaded with weights for the given activity.
*   `_get_activity_label`: Return the human-readable activity label from config.
*   `_get_idle_label`: Return the idle label for a given activity from config.
*   `_detect_best_activity_image`: Run every configured recogniser on an image and pick the best match.
*   `_detect_best_activity_video`: Run every configured activity's VideoProcessor and pick the best match.
*   `index`: No docstring provided.

---
