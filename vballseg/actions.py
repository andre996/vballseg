"""Action spotting: serve / receive / dig / set / attack / block.

1. Touch detection - a contact shows up as a sharp change in the ball's image-space velocity
   (direction and/or speed) with a player within reach.
2. Labelling - volleyball's possession grammar: serve, then each side gets up to three
   contacts (1st = receive/dig, 2nd = set, 3rd or last-before-crossing = attack); a contact at
   the net right after an opponent's attack is a block (which does not count as a contact).

This is a rule-based baseline that needs no training data. `export_training_events` writes the
detections in a simple format you can correct by hand and use to train a learned spotter.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _nearest_player(frame: Dict, xy, ball_r: Optional[float] = None) -> Optional[Dict]:
    """Pick the contacting player. Besides image distance we use apparent size: a ball ~0.21 m
    across next to a ~1.85 m player means box height ~ 8.8 ball diameters at the same depth, which
    separates a big near-side player from a small far-side one that overlap in the image."""
    best, best_c = None, 1e9
    bx, by = xy
    for p in frame["players"]:
        x1, y1, x2, y2 = p["box"]
        hgt = y2 - y1
        # reach zone: box widened a bit and extended upward for jumps / raised arms
        rx1, rx2, ry1, ry2 = x1 - 0.35 * hgt, x2 + 0.35 * hgt, y1 - 0.7 * hgt, y2
        dx = max(rx1 - bx, 0, bx - rx2)
        dy = max(ry1 - by, 0, by - ry2)
        d = np.hypot(dx, dy) / max(hgt, 1)
        if d > 0.25:
            continue
        cost = d + 0.15 * abs(bx - (x1 + x2) / 2) / max(hgt, 1)
        if ball_r and ball_r > 1:
            cost += 0.5 * abs(np.log(hgt / (17.6 * ball_r)))
        if cost < best_c:
            best, best_c = p, cost
    return best


def _ball_radius(frames: List[Dict], i: int, span: int = 6) -> Optional[float]:
    rs = [frames[j]["ball"].get("r", 0) for j in range(max(0, i - span), min(len(frames), i + span + 1))
          if frames[j].get("ball") and frames[j]["ball"].get("r", 0) > 1]
    return float(np.median(rs)) if rs else None


def detect_touches(frames: List[Dict], start: float, end: float, diag: float,
                   min_sep: float = 0.35, thresh: float = 0.55) -> List[Dict]:
    idx = [i for i, f in enumerate(frames) if start <= f["t"] <= end and f.get("ball")]
    if len(idx) < 7:
        return []
    T = np.array([frames[i]["t"] for i in idx])
    P = np.array([frames[i]["ball"]["xy"] for i in idx], float)
    dt = float(np.median(np.diff(T)))
    k = max(2, int(round(0.12 / dt)))
    scores = np.zeros(len(idx))
    for j in range(k, len(idx) - k):
        if T[j + k] - T[j - k] > 6 * k * dt:  # gap in the track
            continue
        v0 = (P[j] - P[j - k]) / (T[j] - T[j - k])
        v1 = (P[j + k] - P[j]) / (T[j + k] - T[j])
        s0, s1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if max(s0, s1) < 0.08 * diag:  # ball nearly static (held) - not a contact
            continue
        cosang = np.dot(v0, v1) / (s0 * s1 + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        ratio = abs(np.log((s1 + 1e-3 * diag) / (s0 + 1e-3 * diag)))
        up_flip = v0[1] > 0.05 * diag and v1[1] < -0.05 * diag  # falling -> rising (image y down)
        scores[j] = ang / 150.0 + 0.35 * min(ratio, 2.0) + (0.3 if up_flip else 0.0)
    touches = []
    order = np.argsort(-scores)
    taken = np.zeros(len(idx), bool)
    for j in order:
        if scores[j] < thresh:
            break
        if taken[max(0, j - 1):j + 2].any():
            continue
        near = np.abs(T - T[j]) < min_sep
        if taken[near].any():
            continue
        f = frames[idx[j]]
        p = _nearest_player(f, P[j], _ball_radius(frames, idx[j]))
        if p is None:
            continue
        taken[j] = True
        touches.append({"t": round(float(T[j]), 2), "frame": f["frame"], "ball_xy": P[j].round(1).tolist(),
                        "player_id": p["id"], "side": p.get("side"), "zone": p.get("zone"),
                        "court_xy": p.get("court"), "score": round(float(scores[j]), 3)})
    touches.sort(key=lambda e: e["t"])
    return touches


def label_touches(touches: List[Dict]) -> List[Dict]:
    out = []
    cur_side, count, last_type, last_t = None, 0, None, -1e9
    for i, e in enumerate(touches):
        e = dict(e)
        side = e.get("side") if e.get("side") in ("A", "B") else None
        nxt_side = None
        if i + 1 < len(touches):
            ns = touches[i + 1].get("side")
            nxt_side = ns if ns in ("A", "B") else None
        conf = min(1.0, 0.4 + 0.3 * e["score"])
        if i == 0:
            typ = "serve"
            cxy = e.get("court_xy")
            if cxy and (cxy[1] < 0.5 or cxy[1] > 17.5):
                conf = min(1.0, conf + 0.25)  # behind the end line
            cur_side, count = side, 0
        elif side is None:
            typ, conf = "touch", 0.3
        else:
            new_poss = side != cur_side
            near_net = e.get("court_xy") is not None and abs(e["court_xy"][1] - 9.0) < 1.5
            if new_poss and last_type == "attack" and near_net and e["t"] - last_t < 0.6:
                typ = "block"
                cur_side, count = side, 0  # block isn't one of the three contacts
            else:
                if new_poss:
                    cur_side, count = side, 1
                else:
                    count += 1
                crosses_next = nxt_side is not None and nxt_side != side
                if count == 1:
                    typ = "receive" if last_type == "serve" else "dig"
                    if crosses_next and i + 1 < len(touches):
                        typ = "free_ball" if last_type != "serve" else "receive"
                elif count == 2:
                    typ = "attack" if crosses_next else "set"
                else:
                    typ = "attack"
                if count > 3:
                    conf *= 0.5  # 4 contacts = likely a missed side change or false touch
        e["type"], e["confidence"] = typ, round(conf, 3)
        out.append(e)
        last_type, last_t = typ, e["t"]
    return out


def spot_actions(frames: List[Dict], rallies: List[Dict], diag: float) -> List[Dict]:
    events = []
    for r in rallies:
        tch = detect_touches(frames, r["core_start"] - 0.5, r["core_end"] + 0.3, diag)
        for e in label_touches(tch):
            e["rally"] = r["id"]
            events.append(e)
    return events
