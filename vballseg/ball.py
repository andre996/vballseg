"""Ball tracking: fuses model detections with a motion-blob fallback.

The COCO 'sports ball' class misses many volleyball frames (motion blur, small size), so we also
look for small, roundish moving blobs and keep the candidate most consistent with a constant-
velocity prediction. After the pass, short gaps are interpolated and outliers removed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


class BallTracker:
    def __init__(self, frame_shape, fps_eff: float, use_motion: bool = True):
        h, w = frame_shape[:2]
        self.diag = float(np.hypot(w, h))
        self.fps = fps_eff
        self.use_motion = use_motion
        self.prev_gray: Optional[np.ndarray] = None
        self.pos: Optional[np.ndarray] = None
        self.vel = np.zeros(2)
        self.miss = 0
        self.hits = 0  # consecutive associations; motion-only tracks must be confirmed
        self.min_area = max(4.0, (0.004 * self.diag) ** 2)
        self.max_area = (0.035 * self.diag) ** 2

    def reset(self):
        self.prev_gray, self.pos, self.vel, self.miss, self.hits = None, None, np.zeros(2), 0, 0

    def _motion_candidates(self, gray: np.ndarray, players: Sequence[Dict], H_cur2prev=None) -> List[Dict]:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return []
        if H_cur2prev is None:  # camera motion unknown -> differencing would light up the background
            return []
        prev = self.prev_gray
        if not np.allclose(H_cur2prev, np.eye(3)):
            prev = cv2.warpPerspective(prev, H_cur2prev, gray.shape[::-1], flags=cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(cv2.GaussianBlur(gray, (3, 3), 0), cv2.GaussianBlur(prev, (3, 3), 0))
        _, m = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_area or a > self.max_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            circ = a / (np.pi * r * r + 1e-6)
            if circ < 0.35:
                continue
            # moving blobs inside a player box are usually limbs: penalise rather than drop
            inside = any(p["box"][0] < x < p["box"][2] and p["box"][1] < y < p["box"][3] for p in players)
            out.append({"xy": [float(x), float(y)], "r": float(r), "inside": inside,
                        "conf": float(0.3 * circ * (0.6 if inside else 1.0)), "src": "motion"})
        return out

    def update(self, gray: np.ndarray, model_dets: List[Dict], players: Sequence[Dict],
               H_cur2prev=np.eye(3)) -> Optional[Dict]:
        cands = list(model_dets)
        if self.use_motion:
            cands += self._motion_candidates(gray, players, H_cur2prev)
        self.prev_gray = gray
        pred = self.pos + self.vel if self.pos is not None else None
        gate = 0.12 * self.diag * (1 + 0.5 * min(self.miss, 6))
        best, best_s = None, -1e9
        for c in cands:
            xy = np.array(c["xy"])
            s = c["conf"] * (2.0 if c["src"] == "model" else 1.0)
            if pred is not None:
                d = np.linalg.norm(xy - pred)
                if d > gate and c["src"] != "model":
                    continue
                if c.get("inside") and (self.hits < 3 or d > 0.4 * gate):
                    continue  # limb motion, unless it continues a confirmed ball track
                s -= 0.8 * d / gate
            elif c["src"] != "model" and (c["conf"] < 0.2 or c.get("inside")):
                continue  # don't start a track from a weak motion blob or a limb
            if s > best_s:
                best, best_s = c, s
        if best is None:
            self.miss += 1
            if self.miss > 8:
                self.pos, self.vel, self.hits = None, np.zeros(2), 0
            return None
        xy = np.array(best["xy"])
        if self.pos is not None:
            self.vel = 0.6 * (xy - self.pos) / max(1, self.miss + 1) + 0.4 * self.vel
            self.hits += 1
        else:
            self.hits = 1
        self.pos, self.miss = xy, 0
        if best["src"] != "model" and (self.hits < 3 or np.linalg.norm(self.vel) < 0.006 * self.diag):
            return None  # unconfirmed or near-static motion track (likely clutter)
        return {"xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)], "conf": round(best["conf"], 3),
                "src": best["src"], "in_player": bool(best.get("inside", False)),
                "r": round(float(best.get("r", 0.0)), 1)}


def _drop_clutter(balls: List[Optional[Dict]], diag: float, win: int) -> List[Optional[Dict]]:
    """A ball in play travels smoothly; limb/clutter tracks wiggle in place, zig-zag between blobs,
    or live inside player boxes. Judge each motion-sourced point by a ~1 s window around it."""
    out = list(balls)
    n = len(balls)
    for i, b in enumerate(balls):
        if not b or b["src"] == "model":
            continue
        W = [balls[j] for j in range(max(0, i - win), min(n, i + win + 1)) if balls[j]]
        if any(x["src"] == "model" for x in W) and len(W) >= 3:
            continue
        P = np.array([x["xy"] for x in W])
        extent = float(np.linalg.norm(P.max(0) - P.min(0))) if len(P) > 1 else 0.0
        path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        inside = np.mean([x.get("in_player", False) for x in W])
        if len(W) < 4 or extent < 0.08 * diag or inside > 0.45 or path > 3.0 * extent:
            out[i] = None
    return out


def clean_trajectory(balls: List[Optional[Dict]], max_gap: int = 6, diag: float = 1000.0) -> List[Optional[Dict]]:
    """Drop clutter tracklets, remove isolated jumps and linearly interpolate short gaps."""
    balls = _drop_clutter(balls, diag, win=max(3, int(2.5 * max_gap)))
    pts = [np.array(b["xy"]) if b else None for b in balls]
    n = len(pts)
    # outliers: point far from both neighbours that are close to each other
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        if a is not None and b is not None and c is not None:
            if np.linalg.norm(b - a) > 0.15 * diag and np.linalg.norm(b - c) > 0.15 * diag \
                    and np.linalg.norm(a - c) < 0.08 * diag:
                pts[i] = None
    out = [dict(b) if (b and pts[i] is not None) else None for i, b in enumerate(balls)]
    i = 0
    while i < n:
        if pts[i] is None:
            j = i
            while j < n and pts[j] is None:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                a, b = pts[i - 1], pts[j]
                for k in range(i, j):
                    f = (k - i + 1) / (j - i + 1)
                    xy = a + f * (b - a)
                    out[k] = {"xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)], "conf": 0.0, "src": "interp"}
            i = j
        else:
            i += 1
    return out
