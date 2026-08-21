"""
Video Processor

Purpose: Video processing pipeline.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Video processing pipeline.

``VideoProcessor.process_video(input_video, output_video)`` processes an
input video frame-by-frame using ``FrameProcessor`` and writes an
annotated output video.  Returns a stats dictionary summarising the run.

Supports multiple activities via the ``activity`` parameter.  Defaults to
``"reinforcement"`` for full backward compatibility.
"""

from __future__ import annotations

import os
import time

import cv2
import yaml

from src.video.frame_processor import FrameProcessor
from src.video.video_utils import open_video, create_video_writer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Process a video file through the pier monitoring pipeline.

    Args:
        activity: Activity key — ``"reinforcement"`` or ``"casting"``.
            Defaults to ``"reinforcement"`` for backward compatibility.
        config_path: Path to the main YAML configuration file.
    """

    def __init__(
        self,
        activity: str = "reinforcement",
        config_path: str = "config.yaml",
    ):
        self.activity = activity
        self.config_path = config_path

        # Load config to resolve activity-specific labels and output dir
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        activities_cfg = config.get("activities", {})
        activity_cfg = activities_cfg.get(activity, {})

        self.activity_label = activity_cfg.get(
            "activity_label", "Pier Stem Reinforcement"
        )
        self.idle_label = activity_cfg.get(
            "idle_label", "Idle / No Significant Activity"
        )
        self.default_output_dir = activity_cfg.get(
            "output_dir", "outputs"
        )

        self.frame_processor = FrameProcessor(
            activity=activity,
            config_path=config_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_video: str,
        output_video: str | None = None,
    ) -> dict:
        """Process a video file and write an annotated output video.

        Args:
            input_video: Path to the source video file.
            output_video: Path for the annotated output video.  When
                ``None`` the file is written to the activity-specific
                output directory defined in config.yaml.

        Returns:
            dict: Processing statistics — frames processed, activity
                frames, idle frames, average confidence, average FPS,
                processing time, output video path, and final activity.
        """
        # --------------------------------------------
        # Resolve output path
        # --------------------------------------------
        if output_video is None:
            output_video = os.path.join(
                self.default_output_dir,
                "pier_monitoring_output.mp4",
            )

        os.makedirs(os.path.dirname(output_video), exist_ok=True)

        # --------------------------------------------
        # Open video
        # --------------------------------------------
        cap, fps, width, height, total_frames = open_video(input_video)

        writer = create_video_writer(output_video, fps, width, height)

        frame_number      = 0
        activity_frames   = 0
        idle_frames       = 0
        confidence_sum    = 0.0

        start_time = time.time()

        print("\n")
        print("=" * 70)
        print(f"VIDEO PROCESSING STARTED  —  activity: {self.activity}")
        print("=" * 70)

        # --------------------------------------------
        # Process every frame
        # --------------------------------------------
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            output = self.frame_processor.process_frame(frame)

            current_activity = output["activity"]
            confidence       = output["confidence"]
            confidence_sum  += confidence

            # ----------------------------------------
            # Activity counter
            # ----------------------------------------
            if current_activity == self.activity_label:
                activity_frames += 1
                color = (0, 255, 0)      # green
            else:
                idle_frames += 1
                color = (0, 0, 255)      # red

            # ----------------------------------------
            # Overlay text
            # ----------------------------------------
            cv2.putText(
                frame,
                f"Activity : {current_activity}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Confidence : {confidence:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Frame : {frame_number}/{total_frames}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

            print(f"Frame {frame_number}/{total_frames} --> {current_activity}")

        # --------------------------------------------
        # Release resources
        # --------------------------------------------
        cap.release()
        writer.release()

        total_time = time.time() - start_time

        average_confidence = (
            confidence_sum / frame_number if frame_number > 0 else 0.0
        )
        average_fps = (
            frame_number / total_time if total_time > 0 else 0.0
        )

        # --------------------------------------------
        # Decide Final Activity
        # --------------------------------------------
        if activity_frames >= idle_frames:
            final_activity = self.activity_label
        else:
            final_activity = self.idle_label

        print("\n")
        print("=" * 70)
        print("VIDEO PROCESSING COMPLETED")
        print("=" * 70)
        print(f"Activity              : {self.activity}")
        print(f"Frames Processed      : {frame_number}")
        print(f"Activity Frames       : {activity_frames}")
        print(f"Idle Frames           : {idle_frames}")
        print(f"Average Confidence    : {average_confidence:.3f}")
        print(f"Average FPS           : {average_fps:.2f}")
        print(f"Processing Time       : {total_time:.2f} seconds")
        print(f"Output Saved          : {output_video}")
        print("\n")
        print("=" * 70)
        print(f"FINAL ACTIVITY : {final_activity}")
        print("=" * 70)

        # --------------------------------------------
        # Return statistics
        # --------------------------------------------
        return {
            "output_video":       output_video,
            "final_activity":     final_activity,
            "activity_label":     self.activity_label,
            "frames_processed":   frame_number,
            "activity_frames":    activity_frames,
            "idle_frames":        idle_frames,
            "average_confidence": round(average_confidence, 3),
            "average_fps":        round(average_fps, 2),
            "processing_time":    round(total_time, 2),
        }