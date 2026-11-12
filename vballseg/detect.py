"""Player + ball detection, instance masks and multi-object tracking (YOLO-seg + ByteTrack, CPU)."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


class Detector:
    """Wraps an Ultralytics segmentation model.

    Works with the stock COCO model (person / sports ball) or a fine-tuned volleyball model whose
    class names contain 'player'/'person' and 'ball'.
    """

    def __init__(self, model_path: str = "yolo11n-seg.pt", imgsz: int = 640, conf: float = 0.25,
                 ball_conf: float = 0.15, keep_masks: bool = True, device: str = "cpu",
                 detect_ball: bool = True):
        from ultralytics import YOLO  # imported lazily so the rest of the package loads without torch

        self.model = YOLO(model_path)
        self.imgsz, self.conf, self.ball_conf = imgsz, conf, ball_conf
        self.keep_masks, self.device = keep_masks, device
        names = self.model.names
        self.person_ids = [i for i, n in names.items() if any(k in n.lower() for k in ("person", "player"))]
        self.ball_ids = [i for i, n in names.items() if "ball" in n.lower()] if detect_ball else []
        if not self.person_ids:
            raise ValueError(f"Model has no person/player class: {names}")
        self.is_seg = getattr(self.model, "task", "") == "segment"

    def __call__(self, frame: np.ndarray) -> Dict:
        res = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz,
            conf=min(self.conf, self.ball_conf), classes=self.person_ids + self.ball_ids,
            device=self.device, verbose=False, retina_masks=False,
        )[0]
        players: List[Dict] = []
        balls: List[Dict] = []
        if res.boxes is None or len(res.boxes) == 0:
            return {"players": players, "ball_dets": balls}
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(boxes)
        polys = res.masks.xy if (self.is_seg and res.masks is not None) else [None] * len(boxes)
        for box, c, k, tid, poly in zip(boxes, confs, clss, ids, polys):
            x1, y1, x2, y2 = [float(v) for v in box]
            if k in self.ball_ids:
                if c >= self.ball_conf:
                    balls.append({"xy": [(x1 + x2) / 2, (y1 + y2) / 2], "r": max(x2 - x1, y2 - y1) / 2,
                                  "conf": float(c), "src": "model"})
                continue
            if c < self.conf:
                continue
            p = {"id": int(tid) if tid is not None else -1, "box": [x1, y1, x2, y2], "conf": float(c),
                 "foot": [(x1 + x2) / 2, y2]}
            if self.keep_masks and poly is not None and len(poly) >= 3:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
                pts = cv2.approxPolyDP(pts, 1.5, True).reshape(-1, 2)
                p["mask"] = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
            players.append(p)
        return {"players": players, "ball_dets": balls}

    def reset_tracks(self) -> None:
        """Call on scene cuts so IDs don't jump across unrelated shots."""
        pred = getattr(self.model, "predictor", None)
        if pred is not None and getattr(pred, "trackers", None):
            for t in pred.trackers:
                t.reset()


def filter_players(players: List[Dict], frame_h: int, court_side_fn=None, max_players: int = 16) -> List[Dict]:
    """Drop tiny detections (crowd far away) and, when court mapping exists, people far off court."""
    if not players:
        return players
    heights = np.array([p["box"][3] - p["box"][1] for p in players])
    ref = np.percentile(heights, 75)
    keep = [p for p, h in zip(players, heights) if h >= 0.35 * ref and h >= 0.03 * frame_h]
    if court_side_fn is not None:
        keep2 = [p for p in keep if court_side_fn(p) != "far_off"]
        if len(keep2) >= 2:
            keep = keep2
    keep.sort(key=lambda p: -p["conf"])
    return keep[:max_players]
