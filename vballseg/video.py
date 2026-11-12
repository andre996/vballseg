"""Video reading that copes with phone clips (variable frame rate, rotation, odd sizes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1 or fps > 480:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()  # read a frame so rotation metadata is applied to the size
    if not ok:
        raise IOError(f"Video has no readable frames: {path}")
    h, w = frame.shape[:2]
    cap.release()
    return VideoInfo(path, float(fps), n, w, h, n / fps if n else 0.0)


def iter_frames(
    path: str, stride: int = 1, max_width: Optional[int] = 960,
    start: float = 0.0, end: Optional[float] = None,
) -> Iterator[Tuple[int, float, np.ndarray, float]]:
    """Yield (frame_index, timestamp_s, frame_bgr_resized, scale).

    Timestamps come from the container (CAP_PROP_POS_MSEC) so VFR phone video stays in sync;
    `scale` maps resized pixel coords back to the original: orig = resized / scale.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    idx = int(round(start * fps)) if start > 0 else 0
    last_t = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and idx > 0 or t <= last_t:  # some containers report 0 / non-monotonic
            t = idx / fps
        last_t = t
        if end is not None and t > end:
            break
        if idx % stride == 0:
            scale = 1.0
            h, w = frame.shape[:2]
            if max_width and w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            yield idx, t, frame, scale
        idx += 1
    cap.release()


def is_scene_cut(prev_gray: np.ndarray, gray: np.ndarray, thresh: float = 0.55) -> bool:
    """Histogram correlation drop => hard cut (mixed/edited clips, replays)."""
    h1 = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < thresh
