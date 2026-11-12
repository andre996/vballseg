"""Annotated overlay rendering: masks, IDs, ball trail, court lines, events, mini-map."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from .court import COURT_L, COURT_W, draw_court

SIDE_COLORS = {"A": (235, 140, 40), "B": (40, 140, 245), None: (170, 170, 170), "far_off": (110, 110, 110)}
EVENT_COLORS = {"serve": (0, 215, 255), "receive": (120, 255, 120), "dig": (120, 255, 120),
                "set": (255, 255, 0), "attack": (60, 60, 255), "block": (255, 80, 255),
                "free_ball": (200, 200, 200), "touch": (200, 200, 200)}


def _minimap(frame: np.ndarray, players: List[Dict], ball_court=None):
    h, w = frame.shape[:2]
    mw = int(min(0.32 * w, 360))
    ppm = mw / (COURT_L + 4)               # pixels per metre (court drawn horizontally)
    mh = int((COURT_W + 4) * ppm)
    x0, y0 = w - mw - 10, h - mh - 10
    roi = frame[y0:y0 + mh, x0:x0 + mw]
    roi[:] = (0.35 * roi + 0.65 * np.array([40, 90, 50])).astype(np.uint8)

    def P(X, Y):  # court (X across, Y along) -> minimap pixel (Y horizontal)
        return int(x0 + (Y + 2) * ppm), int(y0 + (X + 2) * ppm)

    cv2.rectangle(frame, P(0, 0), P(9, 18), (255, 255, 255), 1)
    for Y in (6, 12):
        cv2.line(frame, P(0, Y), P(9, Y), (220, 220, 220), 1)
    cv2.line(frame, P(-0.5, 9), P(9.5, 9), (255, 255, 255), 2)
    for Y0, Y1 in ((0, 9), (9, 18)):
        for X in (3, 6):
            cv2.line(frame, P(X, Y0), P(X, Y1), (130, 160, 130), 1)
    for p in players:
        c = p.get("court")
        if c is None:
            continue
        X, Y = np.clip(c[0], -2, 11), np.clip(c[1], -2, 20)
        cv2.circle(frame, P(X, Y), 5, SIDE_COLORS.get(p.get("side"), SIDE_COLORS[None]), -1, cv2.LINE_AA)
        if p.get("zone"):
            cv2.putText(frame, str(p["zone"]), (P(X, Y)[0] + 6, P(X, Y)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, rec: Dict, trail: List, rally: Optional[Dict], events: List[Dict]) -> np.ndarray:
    out = frame.copy()
    overlay = frame.copy()
    for p in rec["players"]:
        col = SIDE_COLORS.get(p.get("side"), SIDE_COLORS[None])
        if p.get("mask"):
            cv2.fillPoly(overlay, [np.int32(p["mask"])], col)
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    if rec.get("H_court2img") is not None:
        draw_court(out, np.array(rec["H_court2img"]).reshape(3, 3), (0, 230, 255), 1)
    for p in rec["players"]:
        col = SIDE_COLORS.get(p.get("side"), SIDE_COLORS[None])
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
        lab = f"#{p['id']}" + (f" Z{p['zone']}" if p.get("zone") else "")
        cv2.putText(out, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    pts = [np.int32(b) for b in trail if b is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(out, tuple(a), tuple(b), (0, 255, 255), 2, cv2.LINE_AA)
    if rec.get("ball"):
        cv2.circle(out, tuple(int(v) for v in rec["ball"]["xy"]), 8, (0, 255, 255), 2, cv2.LINE_AA)
    # status bar
    live = rally is not None
    cv2.rectangle(out, (8, 8), (210, 38), (0, 0, 0), -1)
    cv2.circle(out, (24, 23), 7, (0, 0, 255) if live else (90, 90, 90), -1)
    cv2.putText(out, f"RALLY {rally['id']}" if live else "dead time", (38, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, e in enumerate(events[:2]):
        txt = f"{e['type'].upper()}  #{e['player_id']} {e.get('side') or ''}"
        cv2.putText(out, txt, (12, 66 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (12, 66 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    EVENT_COLORS.get(e["type"], (255, 255, 255)), 2, cv2.LINE_AA)
    if any(p.get("court") for p in rec["players"]):
        _minimap(out, rec["players"])
    return out
