"""TrackNetV3 ball tracker.

Architecture re-implemented from TrackNetV3 (Chen & Wang, 2023; https://github.com/qaz812345/TrackNetV3,
MIT License, Copyright (c) 2024 qaz812345) so checkpoints trained with that repo load directly:
the upstream `TrackNet_best.pt` (badminton) or volleyball fine-tunes in the same format.

TrackNet looks at a window of consecutive frames (default 8) plus a background (median) image and
outputs one heatmap per frame. This module runs it in streaming fashion: frames are pushed one
by one, and once a window is full it is inferred and the ball positions for those frames are
delivered through a callback. Windows never cross a scene cut.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

TN_H, TN_W = 288, 512


# ----------------------------------------------------------------------------- network
class _Conv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Double(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv_1, self.conv_2 = _Conv(i, o), _Conv(o, o)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class _Triple(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv_1, self.conv_2, self.conv_3 = _Conv(i, o), _Conv(o, o), _Conv(o, o)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    """Same module names as upstream so state_dicts load unchanged."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.down_block_1 = _Double(in_dim, 64)
        self.down_block_2 = _Double(64, 128)
        self.down_block_3 = _Triple(128, 256)
        self.bottleneck = _Triple(256, 512)
        self.up_block_1 = _Triple(768, 256)
        self.up_block_2 = _Double(384, 128)
        self.up_block_3 = _Double(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(self.pool(x1))
        x3 = self.down_block_3(self.pool(x2))
        x = self.bottleneck(self.pool(x3))
        x = self.up_block_1(torch.cat([self.up(x), x3], 1))
        x = self.up_block_2(torch.cat([self.up(x), x2], 1))
        x = self.up_block_3(torch.cat([self.up(x), x1], 1))
        return self.sigmoid(self.predictor(x))


def load_tracknet(path: str, device: str = "cpu") -> Tuple[TrackNet, int, str]:
    """Load an upstream-format checkpoint ({'model', 'param_dict'}) or a bare state_dict."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not isinstance(sd, dict) or "predictor.weight" not in sd:
        raise ValueError(f"{path} does not look like a TrackNet checkpoint")
    seq_len = int(sd["predictor.weight"].shape[0])
    in_dim = int(sd["down_block_1.conv_1.conv.weight"].shape[1])
    bg_mode = None
    if isinstance(ck, dict) and "param_dict" in ck:
        bg_mode = ck["param_dict"].get("bg_mode")
    if bg_mode is None:  # infer from channel count
        bg_mode = {seq_len * 3: "", (seq_len + 1) * 3: "concat", seq_len: "subtract",
                   seq_len * 4: "subtract_concat"}.get(in_dim)
    if bg_mode is None:
        raise ValueError(f"cannot infer background mode (in_dim={in_dim}, seq_len={seq_len})")
    net = TrackNet(in_dim, seq_len)
    net.load_state_dict(sd)
    net.eval().to(device)
    return net, seq_len, bg_mode


# ----------------------------------------------------------------------------- background
def shot_backgrounds(video: str, start: float = 0.0, end: Optional[float] = None,
                     every_s: float = 0.5, max_samples: int = 120) -> List[Dict]:
    """Quick pre-pass: split the video into shots and compute a median background per shot.

    Returns [{"start": t0, "end": t1, "median": RGB uint8 (TN_H, TN_W, 3)}].
    """
    from .video import is_scene_cut

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every_s * fps)))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    i, shots, cur, prev_g, t = 0, [], None, None, start
    while True:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 or (start + i / fps)
        if end is not None and t > end:
            break
        small = cv2.resize(f, (TN_W, TN_H), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if cur is None or (prev_g is not None and is_scene_cut(prev_g, g)):
            if cur is not None:
                cur["end"] = t
                shots.append(cur)
            cur = {"start": t, "end": None, "frames": []}
        cur["frames"].append(small[..., ::-1])  # BGR -> RGB
        prev_g = g
        i += 1
    cap.release()
    if cur is not None:
        cur["end"] = t + 1.0
        shots.append(cur)
    out = []
    for s in shots:
        fr = s.pop("frames")
        if len(fr) > max_samples:
            fr = fr[:: int(np.ceil(len(fr) / max_samples))]
        s["median"] = np.median(np.stack(fr), 0).astype(np.uint8)
        out.append(s)
    return out


# ----------------------------------------------------------------------------- streaming tracker
class TrackNetBall:
    def __init__(self, weights: str, backgrounds: List[Dict], device: str = "cpu", threshold: float = 0.5):
        self.net, self.seq_len, self.bg_mode = load_tracknet(weights, device)
        self.device, self.thr = device, threshold
        self.bgs = backgrounds
        self.buf: List[Tuple[np.ndarray, Callable[[Optional[Dict]], None], Tuple[int, int]]] = []
        self.median: Optional[np.ndarray] = None

    def _bg_for(self, t: float) -> np.ndarray:
        for s in self.bgs:
            if s["start"] - 0.3 <= t < s["end"]:
                return s["median"]
        return self.bgs[-1]["median"] if self.bgs else np.zeros((TN_H, TN_W, 3), np.uint8)

    def push(self, frame_bgr: np.ndarray, t: float, deliver: Callable[[Optional[Dict]], None]) -> None:
        """Queue a frame; `deliver(ball_or_None)` is called once its window has been inferred."""
        if not self.buf:
            self.median = self._bg_for(t)
        rgb = cv2.resize(frame_bgr, (TN_W, TN_H), interpolation=cv2.INTER_AREA)[..., ::-1]
        self.buf.append((rgb, deliver, frame_bgr.shape[:2]))
        if len(self.buf) == self.seq_len:
            self.flush()

    def flush(self) -> None:
        """Infer whatever is buffered (pads short windows by repeating the last frame)."""
        if not self.buf:
            return
        n = len(self.buf)
        frames = [b[0] for b in self.buf] + [self.buf[-1][0]] * (self.seq_len - n)
        med = self.median.astype(np.float32)
        chans = []
        if self.bg_mode == "concat":
            chans.append(np.moveaxis(med, -1, 0))
        for f in frames:
            f = f.astype(np.float32)
            if self.bg_mode == "subtract":
                chans.append(np.abs(f - med).sum(2, keepdims=True).clip(0, 255).transpose(2, 0, 1))
            elif self.bg_mode == "subtract_concat":
                chans.append(np.moveaxis(f, -1, 0))
                chans.append(np.abs(f - med).sum(2, keepdims=True).clip(0, 255).transpose(2, 0, 1))
            else:
                chans.append(np.moveaxis(f, -1, 0))
        x = torch.from_numpy(np.concatenate(chans, 0)[None] / 255.0).float().to(self.device)
        with torch.no_grad():
            hm = self.net(x)[0].cpu().numpy()  # (seq_len, H, W)
        for k, (_, deliver, (h, w)) in enumerate(self.buf):
            deliver(self._locate(hm[k], w / TN_W, h / TN_H))
        self.buf = []

    def _locate(self, heat: np.ndarray, sx: float, sy: float) -> Optional[Dict]:
        m = (heat > self.thr).astype(np.uint8)
        if not m.any():
            return None
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        conf = float(heat[y:y + bh, x:x + bw].max())
        return {"xy": [round((x + bw / 2) * sx, 1), round((y + bh / 2) * sy, 1)], "conf": round(conf, 3),
                "src": "tracknet", "in_player": False, "r": 0.0}
