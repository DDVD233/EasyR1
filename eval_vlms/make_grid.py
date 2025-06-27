#!/usr/bin/env python3
"""
Uniformly sample 4 frames from every .mp4 file inside the `images/` directory
and save them as a single 2×2 grid JPEG next to the original video.

Requirements
------------
pip install opencv-python numpy

Directory layout
----------------
project_root/
├── images/
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── ...
└── sample_frames.py   ← (this script)
"""

from pathlib import Path

import cv2
import numpy as np


# Root folder that contains the videos
ROOT = Path("images")


def sample_four_frames(video_path: Path) -> list[np.ndarray]:
    """Return 4 uniformly spaced frames (BGR images) from the given video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError(f"No frames found in {video_path}")

    # Frame indices spaced uniformly across the clip
    idxs = np.linspace(0, total - 1, num=4, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IOError(f"Could not read frame {idx} from {video_path}")
        frames.append(frame)
    cap.release()
    return frames


def make_grid(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange 4 same-size frames into a 2×2 grid (top-left, top-right, bottom-left, bottom-right)."""
    h, w = frames[0].shape[:2]

    # Ensure all frames are the same size (resize if clip contains variable resolutions)
    resized = [cv2.resize(f, (w, h)) for f in frames]

    top = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:])
    grid = np.vstack((top, bottom))
    return grid


def main():
    mp4_files = sorted(ROOT.rglob("*.mp4"))

    if not mp4_files:
        print("No .mp4 files found under", ROOT.resolve())
        return

    for vid in mp4_files:
        try:
            frames = sample_four_frames(vid)
            grid = make_grid(frames)

            out_path = vid.with_suffix(".jpg")
            cv2.imwrite(str(out_path), grid)  # Saves in BGR → JPEG
            print(f"Saved {out_path.relative_to(ROOT.parent)}")
        except Exception as e:
            print(f"⚠️  {vid}: {e}")


if __name__ == "__main__":
    main()
