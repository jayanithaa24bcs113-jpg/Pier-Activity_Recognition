"""
Video Utils

Purpose: Video utility helpers.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Video utility helpers.

Provide `open_video` and `create_video_writer` helpers used by the
video processing pipeline.
"""

import cv2


def open_video(video_path):
    """
    Open a video.

    Returns:
        cap
        fps
        width
        height
        total_frames
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, width, height, total_frames


def create_video_writer(output_path, fps, width, height):
    """
    Create output video writer.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(

        output_path,

        fourcc,

        fps,

        (width, height)

    )

    return writer