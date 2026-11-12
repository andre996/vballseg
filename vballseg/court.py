"""Court detection, image<->court homography, and zone mapping.

Court coordinates (metres): X across the court 0..9, Y along it 0..18, net at Y = 9.
Side "A" is Y < 9 (nearer the camera for end-view footage, left side for side-view footage).

Zones follow standard rotation numbering, seen by each team facing the net:
    front row: 4 3 2      back row: 5 6 1     (1 = back-right, the server's zone)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COURT_W, COURT_L, NET_Y, ATTACK = 9.0, 18.0, 9.0, 3.0
CORNERS = np.float32([[0, 0], [9, 0], [9, 18], [0, 18]])  # near-left, near-right, far-right, far-left
POINT_PRESETS = {
    "corners": CORNERS,
    # far half only (near baseline out of shot): centre-line ends + far baseline corners
    "far_half": np.float32([[0, 9], [9, 9], [9, 18], [0, 18]]),
    # near half only (far end hidden / zoomed in)
    "near_half": np.float32([[0, 0], [9, 0], [9, 9], [0, 9]]),
}


@dataclass
class CourtModel:
    H_img2court: np.ndarray  # in the *reference* frame of the shot
    source: str = "auto"      # auto | manual
    confidence: float = 0.0
    ref_time: float = 0.0
    img_pts: List[List[float]] = field(default_factory=list)
    court_pts: List[List[float]] = field(default_factory=list)

    @property
    def H_court2img(self) -> np.ndarray:
        return np.linalg.inv(self.H_img2court)

    def to_dict(self) -> Dict:
        return {"H_img2court": self.H_img2court.tolist(), "source": self.source,
                "confidence": round(self.confidence, 3), "ref_time": self.ref_time,
                "img_pts": self.img_pts, "court_pts": self.court_pts}

    @staticmethod
    def from_points(img_pts, court_pts, source="manual", ref_time=0.0) -> "CourtModel":
        img_pts, court_pts = np.float32(img_pts), np.float32(court_pts)
        H, _ = cv2.findHomography(img_pts, court_pts, 0)
        if H is None:
            raise ValueError("Degenerate calibration points")
        return CourtModel(H, source, 1.0 if source == "manual" else 0.5, ref_time,
                          img_pts.tolist(), court_pts.tolist())

    @staticmethod
    def load(path: str, scale: float = 1.0) -> "CourtModel":
        """Load a calibration JSON (from `vballseg calibrate`). `scale` = analysis resize factor."""
        d = json.load(open(path))
        img = np.float32(d["img_pts"]) * scale
        return CourtModel.from_points(img, d["court_pts"], "manual", d.get("ref_time", 0.0))


def project(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def zone_of(X: float, Y: float, margin: float = 1.0) -> Dict:
    """Map a court position to side / row / zone. Off-court players get zone 0 (free zone)."""
    if X < -margin - 3 or X > COURT_W + margin + 3 or Y < -margin - 6 or Y > COURT_L + margin + 6:
        return {"side": "far_off", "zone": None, "row": None}
    side = "A" if Y < NET_Y else "B"
    inside = -margin <= X <= COURT_W + margin and -margin <= Y <= COURT_L + margin
    d_net = abs(Y - NET_Y)
    row = "front" if d_net <= ATTACK else "back"
    # 'right' for team A (facing +Y) is high X; for team B (facing -Y) it is low X
    xr = X if side == "A" else COURT_W - X
    col = 0 if xr < 3 else (1 if xr < 6 else 2)  # 0 = left, 1 = middle, 2 = right
    zone = {("front", 0): 4, ("front", 1): 3, ("front", 2): 2,
            ("back", 0): 5, ("back", 1): 6, ("back", 2): 1}[(row, col)]
    return {"side": side, "zone": zone if inside else 0, "row": row}


# ----------------------------------------------------------------------------- auto detection
def _line_mask(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = max(9, int(frame.shape[1] / 60) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    _, m = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m[(hsv[..., 1] > 80)] = 0  # court lines are white-ish (low saturation)
    m[tophat < 25] = 0
    return m


def _segments_to_lines(segs, w, h):
    lines = []
    for x1, y1, x2, y2 in segs:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        L = np.hypot(x2 - x1, y2 - y1)
        horiz = ang < 30 or ang > 150
        if horiz:  # y at image centre x
            t = (w / 2 - x1) / (x2 - x1 + 1e-9)
            icpt = y1 + t * (y2 - y1)
        else:      # x at image centre y
            t = (h / 2 - y1) / (y2 - y1 + 1e-9)
            icpt = x1 + t * (x2 - x1)
        a = ang if ang <= 90 else ang - 180  # continuous around 0 for horizontals
        lines.append({"seg": (x1, y1, x2, y2), "ang": a if horiz else ang, "icpt": icpt, "len": L, "horiz": horiz})
    return lines


def _merge(lines, icpt_tol, ang_tol=6.0):
    lines = sorted(lines, key=lambda l: l["icpt"])
    groups: List[List[Dict]] = []
    for l in lines:
        for g in groups:
            ref = max(g, key=lambda q: q["len"])
            if abs(l["icpt"] - ref["icpt"]) < icpt_tol and abs(l["ang"] - ref["ang"]) < ang_tol:
                g.append(l)
                break
        else:
            groups.append([l])
    merged = []
    for g in groups:
        pts = np.float32([[s["seg"][0], s["seg"][1]] for s in g] + [[s["seg"][2], s["seg"][3]] for s in g])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        merged.append({"p": np.array([x0, y0]), "d": np.array([vx, vy]),
                       "icpt": float(np.average([s["icpt"] for s in g], weights=[s["len"] for s in g])),
                       "len": float(sum(s["len"] for s in g))})
    return [m for m in merged if m["len"] > 0]


def _intersect(l1, l2):
    A = np.array([l1["d"], -l2["d"]]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, l2["p"] - l1["p"])
    return l1["p"] + t[0] * l1["d"]


def detect_court(frame: np.ndarray, ref_time: float = 0.0) -> Optional[CourtModel]:
    """Best-effort automatic court detection from painted lines.

    Works when both baselines and both sidelines are visible. Returns None (or low confidence)
    otherwise -- use `vballseg calibrate` for those clips.
    """
    h, w = frame.shape[:2]
    m = _line_mask(frame)
    segs = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=60, minLineLength=int(0.12 * w), maxLineGap=int(0.02 * w))
    if segs is None or len(segs) < 4:
        return None
    lines = _segments_to_lines(segs.reshape(-1, 4).astype(float), w, h)
    hz = _merge([l for l in lines if l["horiz"]], 0.025 * h)
    st = _merge([l for l in lines if not l["horiz"]], 0.025 * w)
    hz = [l for l in hz if l["len"] > 0.15 * w]
    st = [l for l in st if l["len"] > 0.08 * h]
    if len(hz) < 2 or len(st) < 2:
        return None
    # the family with more parallel lines runs parallel to the net (baselines, attack lines, centre)
    end_view = len(hz) >= len(st)
    cross, long_ = (hz, st) if end_view else (st, hz)
    cross = sorted(cross, key=lambda l: l["icpt"])
    long_ = sorted(long_, key=lambda l: l["icpt"])
    if end_view:
        near_base, far_base = cross[-1], cross[0]          # bottom of image is nearer
        side0, side9 = long_[0], long_[-1]                 # left sideline = X 0
    else:
        near_base, far_base = cross[0], cross[-1]          # left of image = side A
        side0, side9 = long_[0], long_[-1]                 # top sideline (far) = X 0
    pts = [_intersect(near_base, side0), _intersect(near_base, side9),
           _intersect(far_base, side9), _intersect(far_base, side0)]
    if any(p is None for p in pts):
        return None
    quad = np.float32(pts)
    pad = 0.25
    if (quad[:, 0] < -pad * w).any() or (quad[:, 0] > (1 + pad) * w).any() or \
            (quad[:, 1] < -pad * h).any() or (quad[:, 1] > (1 + pad) * h).any():
        return None
    area = cv2.contourArea(quad)
    if area < 0.08 * w * h or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    cm = CourtModel.from_points(quad, CORNERS, "auto", ref_time)
    # confidence: does a detected cross-line sit where the centre line / attack lines should be?
    H_c2i = cm.H_court2img
    hits = 0
    for Yl in (6.0, 9.0, 12.0):
        a, b = project(H_c2i, [[0, Yl], [9, Yl]])
        mid = (a + b) / 2
        dmin = min(abs(l["d"][0] * (mid - l["p"])[1] - l["d"][1] * (mid - l["p"])[0]) for l in cross)
        hits += dmin < 0.03 * max(w, h)
    cm.confidence = 0.4 + 0.2 * hits
    return cm


def draw_court(frame: np.ndarray, H_court2img: np.ndarray, color=(0, 255, 255), thickness=2):
    segs = [([0, 0], [9, 0]), ([9, 0], [9, 18]), ([9, 18], [0, 18]), ([0, 18], [0, 0]),
            ([0, 9], [9, 9]), ([0, 6], [9, 6]), ([0, 12], [9, 12])]
    for a, b in segs:
        pa, pb = project(H_court2img, [a, b])
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)) and np.abs(np.r_[pa, pb]).max() < 1e5:
            cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb), color,
                     thickness + (1 if a[1] == 9 and b[1] == 9 else 0), cv2.LINE_AA)
