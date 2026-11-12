"""Camera-motion compensation for handheld / panning phone footage.

Estimates a homography from each frame to a reference frame using ORB features on the
background (people masked out), chaining frame-to-frame estimates. The court calibration is
done once in the reference frame and carried through the shot with this transform.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, n_features: int = 1500, still_thresh_px: float = 1.0):
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev = None  # (keypoints, descriptors)
        self.H_to_ref = np.eye(3)
        self.last_H = np.eye(3)  # current -> previous frame
        self.ok = True
        self.still_thresh = still_thresh_px

    def reset(self):
        self.prev, self.H_to_ref, self.ok, self.last_H = None, np.eye(3), True, np.eye(3)

    def update(self, gray: np.ndarray, players: Sequence[dict] = ()) -> np.ndarray:
        """Return H mapping current-frame pixels -> reference-frame pixels."""
        mask = np.full(gray.shape, 255, np.uint8)
        for p in players:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            mask[max(0, y1):y2, max(0, x1):x2] = 0
        kp, des = self.orb.detectAndCompute(gray, mask)
        if self.prev is None or des is None or self.prev[1] is None:
            self.prev = (kp, des)
            return self.H_to_ref
        kp0, des0 = self.prev
        matches = self.bf.match(des, des0) if len(kp) >= 8 and len(kp0) >= 8 else []
        H = None
        if len(matches) >= 15:
            src = np.float32([kp[m.queryIdx].pt for m in matches])
            dst = np.float32([kp0[m.trainIdx].pt for m in matches])
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is not None and inl is not None and inl.sum() < 12:
                H = None
        self.ok = H is not None
        self.last_H = H  # exact frame-to-frame motion (used for motion-compensated differencing)
        if H is not None:
            shift = np.median(np.linalg.norm(src - dst, axis=1))
            if shift >= self.still_thresh:  # tripod: don't accumulate sub-pixel jitter into the chain
                self.H_to_ref = self.H_to_ref @ H
        self.prev = (kp, des)
        return self.H_to_ref
