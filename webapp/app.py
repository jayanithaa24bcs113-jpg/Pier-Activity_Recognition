"""
App

Purpose: Flask webapp front-end for Pier Monitoring demo.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Flask webapp front-end for Pier Monitoring demo.

Handles image and video uploads, runs the recognition pipeline across
ALL configured activities, and automatically determines which activity
is occurring — the user does not select an activity.

Supports any number of activities defined under ``activities`` in
config.yaml (currently Pier Stem Reinforcement, Pier Stem Casting,
Pier Cap Reinforcement, and Pier Cap Casting; designed to scale to
additional stages without code changes).

Pier Cap Reinforcement (Stage 3) and Pier Cap Casting (Stage 4)
intentionally have no UI selector — they participate in auto-detection
like the other stages, but while their models are untrained
(``model_trained: false`` in config.yaml) they never win the
auto-detect comparison, since their recognisers return "Model Not
Trained" for every image. Informational banners are shown instead, and
disappear automatically once training completes and the respective
flags are flipped to ``true`` — no other code change required.

Cross-stage confidence comparison: each recogniser's rules score on a
different numeric scale (e.g. Stage 3's composite rules score up to 5
points per rule with a fire-threshold of 7, while Stage 2/4 score in
the 0.4-1.0 range with fire-thresholds around 1.2). Comparing raw
smoothed scores directly across stages would let whichever stage has
the largest numeric scale dominate the "best match" comparison
regardless of which stage is genuinely more confident. To avoid this,
confidence is computed as ``total_smoothed_score / normalization_threshold``
for the winning recogniser — a ratio expressing "how far past my own
firing threshold am I", which is comparable across stages with very
different internal scoring scales.
"""

from __future__ import annotations

import os
import sys
import shutil

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

# -------------------------------------------------------

from flask import Flask, render_template, request
from ultralytics import YOLO
import yaml

from src.activity.activity_recognizer import (
    ActivityRecognizer,
    CastingActivityRecognizer,
    CapReinforcementActivityRecognizer,
    CapCastingActivityRecognizer,
)
from src.video.video_processor import VideoProcessor

app = Flask(__name__)

# -------------------------------------------------------
# Folders
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
RESULT_FOLDER = os.path.join(STATIC_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------------------------------------------------------
# Load Config
# -------------------------------------------------------

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

# -------------------------------------------------------
# Registry of recogniser classes per activity key.
# Add new stages here as they are built — everything else
# (detection loop, best-match selection, UI) adapts automatically.
# -------------------------------------------------------

RECOGNIZER_REGISTRY = {
    "reinforcement": ActivityRecognizer,
    "casting": CastingActivityRecognizer,
    "cap_reinforcement": CapReinforcementActivityRecognizer,
    "cap_casting": CapCastingActivityRecognizer,
}

# -------------------------------------------------------
# Stage 3 / Stage 4 availability flags — computed once at startup.
#
# When False, the stage still participates in auto-detection (it simply
# never wins, since its recogniser reports "Model Not Trained" for
# every image) and the template shows an informational banner. No
# radio button or upload gating is added — Stage 1 and Stage 2 continue
# working exactly as before regardless of these flags.
# -------------------------------------------------------

CAP_REINFORCEMENT_AVAILABLE = bool(
    _config.get("activities", {})
    .get("cap_reinforcement", {})
    .get("model_trained", False)
)

CAP_CASTING_AVAILABLE = bool(
    _config.get("activities", {})
    .get("cap_casting", {})
    .get("model_trained", False)
)

# -------------------------------------------------------
# Supported Extensions
# -------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _configured_activities() -> list:
    """Return the list of activity keys defined in config.yaml.

    Only keys that also have a registered recogniser class in
    ``RECOGNIZER_REGISTRY`` are returned, so partially-configured or
    not-yet-implemented stages are skipped safely.

    Returns:
        list[str]: Activity keys, e.g. ["reinforcement", "casting",
            "cap_reinforcement", "cap_casting"].
    """
    activities_cfg = _config.get("activities", {})
    return [
        key for key in activities_cfg.keys()
        if key in RECOGNIZER_REGISTRY
    ]


def _build_recognizer(activity: str):
    """Instantiate the recogniser for a given activity key.

    Args:
        activity: Activity key, e.g. "reinforcement", "casting",
            "cap_reinforcement", or "cap_casting".

    Returns:
        An instance of the registered recogniser class for this activity.
    """
    recognizer_cls = RECOGNIZER_REGISTRY[activity]
    return recognizer_cls(CONFIG_PATH)


def _build_detector(activity: str) -> YOLO | None:
    """Return a YOLO model loaded with weights for the given activity.

    Placeholder-safe: if the resolved weights file does not exist on
    disk (e.g. Stage 3/4 before training), returns ``None`` instead of
    raising. Callers must check for ``None`` before use.

    Args:
        activity: Activity key.

    Returns:
        YOLO: Loaded model instance, or ``None`` if weights are missing.
    """
    activities_cfg = _config.get("activities", {})
    activity_cfg = activities_cfg.get(activity, {})
    weights = activity_cfg.get(
        "weights",
        _config["model"]["weights"],  # legacy fallback
    )

    if not os.path.exists(weights):
        return None

    return YOLO(weights)


def _get_activity_label(activity: str) -> str:
    """Return the human-readable activity label from config.

    Args:
        activity: Activity key.

    Returns:
        str: Display label, e.g. "Pier Stem Casting".
    """
    activities_cfg = _config.get("activities", {})
    activity_cfg = activities_cfg.get(activity, {})
    return activity_cfg.get("activity_label", activity)


def _get_idle_label(activity: str) -> str:
    """Return the idle label for a given activity from config.

    Args:
        activity: Activity key.

    Returns:
        str: Idle label string for this activity.
    """
    activities_cfg = _config.get("activities", {})
    activity_cfg = activities_cfg.get(activity, {})
    return activity_cfg.get("idle_label", f"Idle / No {activity} Activity")


def _detect_best_activity_image(image_path: str) -> dict:
    """Run every configured recogniser on an image and pick the best match.

    Each activity's recogniser is run independently. Any recogniser that
    returns its *non-idle* label is a candidate. Among candidates, the
    one with the highest NORMALIZED confidence score wins — normalized
    confidence is ``total_smoothed_score / recognizer.normalization_threshold``,
    a ratio expressing how far past that stage's own firing threshold
    it landed. This keeps the comparison fair across stages whose rules
    score on very different numeric scales (see module docstring). If
    no recogniser reports its activity (all idle), a generic idle result
    is returned.

    Stages 3 and 4 participate like any other activity: while untrained,
    their recognisers return "Model Not Trained" for every image, which
    never equals their target labels, so they simply never become
    candidates — no special-casing required here.

    Args:
        image_path: Path to the uploaded image.

    Returns:
        dict: Keys ``winning_activity`` (key string or None),
            ``activity_label`` (display string), ``confidence`` (float,
            normalized ratio — NOT a raw rule score), ``attributes``,
            ``raw_scores``, ``smoothed_scores`` (from the winning
            recogniser, or empty if none won), ``all_results``
            (per-activity breakdown for transparency, also using
            normalized confidence).
    """
    all_results = {}
    best_activity = None
    best_confidence = -1.0
    best_output = None

    for activity in _configured_activities():

        recognizer = _build_recognizer(activity)
        output = recognizer.recognize(image_path)

        predicted_label = output["activity"]
        idle_label = _get_idle_label(activity)
        target_label = _get_activity_label(activity)

        # Normalized confidence: total smoothed score relative to this
        # stage's own firing threshold, so stages with different rule
        # scoring scales are directly comparable. A value of 1.0 means
        # "exactly at my own threshold"; 1.5 means "50% past it".
        raw_total = sum(output["smoothed_scores"].values())
        normalization_threshold = getattr(
            recognizer, "normalization_threshold", 1.0
        ) or 1.0  # guard against a zero/None threshold causing ZeroDivisionError

        confidence = (
            round(raw_total / normalization_threshold, 3)
            if output["smoothed_scores"]
            else 0.0
        )

        all_results[activity] = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_match": predicted_label == target_label,
        }

        # Only candidates that actually fired their own activity compete
        if predicted_label == target_label:
            if confidence > best_confidence:
                best_confidence = confidence
                best_activity = activity
                best_output = output

    if best_activity is None:
        return {
            "winning_activity": None,
            "activity_label": "Idle / No Activity Detected",
            "confidence": 0.0,
            "attributes": {},
            "raw_scores": {},
            "smoothed_scores": {},
            "all_results": all_results,
        }

    return {
        "winning_activity": best_activity,
        "activity_label": _get_activity_label(best_activity),
        "confidence": round(best_confidence, 3),
        "attributes": best_output["attributes"],
        "raw_scores": best_output["raw_scores"],
        "smoothed_scores": best_output["smoothed_scores"],
        "all_results": all_results,
    }


def _detect_best_activity_video(input_path: str, output_path: str) -> dict:
    """Run every configured activity's VideoProcessor and pick the best match.

    Each activity's video processor runs over the full video independently.
    The activity whose frames most often match its own label (i.e. fewest
    idle frames, by ratio) and with the highest average confidence wins.
    Its annotated output video is what gets shown to the user.

    Note: video scoring uses ``average_confidence`` from
    ``VideoProcessor``/``FrameProcessor``, which is derived from raw
    smoothed scores per frame (max value), not the normalized ratio used
    in ``_detect_best_activity_image``. This is an existing, separate
    scoring path for video and is intentionally left as-is here — the
    cross-stage raw-score-scale issue applies to it too in principle,
    but is out of scope for this fix since it was not the reported bug.

    Args:
        input_path: Path to the uploaded video.
        output_path: Base path for the winning annotated output video.

    Returns:
        dict: Keys ``winning_activity``, ``activity_label``,
            ``stats`` (the winning VideoProcessor stats dict),
            ``all_results`` (per-activity summary for transparency).
    """
    all_results = {}
    best_activity = None
    best_score = -1.0
    best_stats = None

    for activity in _configured_activities():

        processor = VideoProcessor(activity=activity, config_path=CONFIG_PATH)

        # Each activity gets its own temp output path; only the winner's
        # file is kept and copied to the final output_path.
        candidate_output = output_path.replace(
            ".mp4", f"_{activity}_candidate.mp4"
        )

        stats = processor.process_video(
            input_video=input_path,
            output_video=candidate_output,
        )

        target_label = _get_activity_label(activity)
        matched = stats["final_activity"] == target_label

        # Score: fraction of frames matching this activity, weighted by
        # average confidence — favors both consistency and certainty.
        total_frames = max(stats["frames_processed"], 1)
        frame_ratio = stats["activity_frames"] / total_frames
        score = frame_ratio * stats["average_confidence"]

        all_results[activity] = {
            "final_activity": stats["final_activity"],
            "activity_frames": stats["activity_frames"],
            "idle_frames": stats["idle_frames"],
            "average_confidence": stats["average_confidence"],
            "is_match": matched,
        }

        if matched and score > best_score:
            best_score = score
            best_activity = activity
            best_stats = stats

        else:
            # Clean up non-winning candidate video to save disk space
            if os.path.exists(candidate_output):
                os.remove(candidate_output)

    if best_activity is None:
        return {
            "winning_activity": None,
            "activity_label": "Idle / No Activity Detected",
            "stats": None,
            "all_results": all_results,
        }

    # Promote the winning candidate file to the final expected path
    winning_candidate_path = best_stats["output_video"]
    if os.path.exists(winning_candidate_path):
        shutil.move(winning_candidate_path, output_path)
        best_stats["output_video"] = output_path

    return {
        "winning_activity": best_activity,
        "activity_label": _get_activity_label(best_activity),
        "stats": best_stats,
        "all_results": all_results,
    }

# -------------------------------------------------------
# Home Page
# -------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    uploaded_image = None
    result_image = None
    uploaded_video = None
    result_video = None
    activity = None
    activity_display_label = None
    confidence = None
    detections = []
    attributes = {}
    raw_scores = {}
    smoothed_scores = {}
    video_stats = None
    all_activity_results = {}
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "No file selected."

        else:

            file = request.files["image"]

            if file.filename == "":
                error = "Please choose a file."

            else:

                extension = os.path.splitext(file.filename)[1].lower()

                upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(upload_path)

                # ===========================================
                # IMAGE — auto-detect activity
                # ===========================================

                if extension in IMAGE_EXTENSIONS:

                    detection_result = _detect_best_activity_image(upload_path)

                    winning_activity = detection_result["winning_activity"]
                    activity = detection_result["activity_label"]
                    activity_display_label = detection_result["activity_label"]
                    confidence = detection_result["confidence"]
                    attributes = detection_result["attributes"]
                    raw_scores = detection_result["raw_scores"]
                    smoothed_scores = detection_result["smoothed_scores"]
                    all_activity_results = detection_result["all_results"]

                    uploaded_image = "uploads/" + file.filename

                    # Run the winning activity's detector for the
                    # annotated bounding-box image. Falls back to the
                    # first configured activity if nothing matched, so
                    # the user still sees raw detections.
                    detector_activity = (
                        winning_activity
                        or (_configured_activities()[0]
                            if _configured_activities() else None)
                    )

                    detector = (
                        _build_detector(detector_activity)
                        if detector_activity
                        else None
                    )

                    if detector is not None:

                        results = detector.predict(
                            source=upload_path,
                            save=True,
                            conf=0.25,
                            project="webapp/static",
                            name="predict",
                            exist_ok=True,
                            device="cpu",
                        )

                        prediction_path = os.path.join(
                            results[0].save_dir, file.filename
                        )
                        final_path = os.path.join(RESULT_FOLDER, file.filename)
                        shutil.copy(prediction_path, final_path)

                        result_image = "results/" + file.filename

                        detections = []
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            detections.append({
                                "name": detector.names[cls],
                                "confidence": round(float(box.conf[0]), 3),
                            })

                # ===========================================
                # VIDEO — auto-detect activity
                # ===========================================

                elif extension in VIDEO_EXTENSIONS:

                    output_name = (
                        os.path.splitext(file.filename)[0] + "_output.mp4"
                    )
                    output_path = os.path.join(RESULT_FOLDER, output_name)

                    detection_result = _detect_best_activity_video(
                        upload_path, output_path
                    )

                    activity = detection_result["activity_label"]
                    activity_display_label = detection_result["activity_label"]
                    all_activity_results = detection_result["all_results"]

                    uploaded_video = "uploads/" + file.filename

                    stats = detection_result["stats"]
                    if stats is not None:
                        result_video = "results/" + output_name
                        video_stats = stats
                        confidence = stats["average_confidence"]
                    else:
                        confidence = 0.0

                # ===========================================
                # Unsupported File
                # ===========================================

                else:
                    error = "Unsupported file type."

    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        result_image=result_image,
        uploaded_video=uploaded_video,
        result_video=result_video,
        detections=detections,
        activity=activity,
        activity_display_label=activity_display_label,
        confidence=confidence,
        attributes=attributes,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        video_stats=video_stats,
        all_activity_results=all_activity_results,
        cap_reinforcement_available=CAP_REINFORCEMENT_AVAILABLE,
        cap_casting_available=CAP_CASTING_AVAILABLE,
        error=error,
    )

# -------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)