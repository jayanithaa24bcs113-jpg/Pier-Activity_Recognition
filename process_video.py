"""
Process Video

Purpose: CLI front-end to process a single input video.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

CLI front-end to process a single input video.

Runs full pier monitoring recognition over every frame of a video and
writes an annotated output video.

Usage
-----
    python process_video.py --activity casting --source path/to/video.mp4
    python process_video.py --activity cap_reinforcement --source path/to/video.mp4
    python process_video.py --activity cap_casting --source path/to/video.mp4

When ``--activity`` is omitted it defaults to ``reinforcement`` for full
backward compatibility. When ``--source`` is omitted the script falls
back to an interactive prompt, matching the original behaviour.

Placeholder-safe: for ``cap_reinforcement`` or ``cap_casting`` (Stages 3
and 4), if the model has not been trained yet, every frame's recognition
returns "Model Not Trained" and the video is processed and saved
normally, with the final activity reported as the idle label — no crash.
"""

import argparse
import os

from src.video.video_processor import VideoProcessor


def main():
    """Parse CLI args (or prompt), process the video, print summary."""

    parser = argparse.ArgumentParser(
        description="Process a video through the pier monitoring pipeline."
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="reinforcement",
        choices=["reinforcement", "casting", "cap_reinforcement", "cap_casting"],
        help="Activity to run (default: reinforcement).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the input video. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve input video — flag takes priority, else prompt
    # (preserves original interactive behaviour)
    # --------------------------------------------------------
    video_path = args.source

    if video_path is None:
        video_path = input("Enter video path : ").strip()

    if not os.path.exists(video_path):
        print("\nVideo not found!")
        return

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------
    processor = VideoProcessor(
        activity=args.activity,
        config_path=args.config,
    )

    processor.process_video(
        input_video=video_path,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()