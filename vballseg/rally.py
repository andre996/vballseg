"""Rally segmentation: find when play is live vs. dead time.

Signals (all scale-invariant so they work across phone clips with different zoom):
  * ball activity  - ball visible and moving fast relative to the frame diagonal
  * player motion  - tracked players' foot displacement in body-heights per second
  * referee whistle (optional) - tonal energy in the 2-4.5 kHz band of the audio track
"""
from __future__ import annotations

import subprocess
from typing import Dict, List, Optional

import numpy as np


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def activity_signals(frames: List[Dict], diag: float) -> Dict[str, np.ndarray]:
    t = np.array([f["t"] for f in frames])
    n = len(frames)
    ball_speed = np.zeros(n)
    ball_seen = np.zeros(n)
    motion = np.zeros(n)
    for i in range(1, n):
        dt = max(1e-3, t[i] - t[i - 1])
        b0, b1 = frames[i - 1].get("ball"), frames[i].get("ball")
        if b1:
            ball_seen[i] = 1.0
        if b0 and b1:
            ball_speed[i] = np.hypot(b1["xy"][0] - b0["xy"][0], b1["xy"][1] - b0["xy"][1]) / diag / dt
        prev = {p["id"]: p for p in frames[i - 1]["players"] if p["id"] >= 0}
        sp = []
        for p in frames[i]["players"]:
            q = prev.get(p["id"])
            if q is None:
                continue
            hgt = max(10.0, p["box"][3] - p["box"][1])
            sp.append(np.hypot(p["foot"][0] - q["foot"][0], p["foot"][1] - q["foot"][1]) / hgt / dt)
        if sp:
            # mean of the top half: during a rally most players move; in dead time a few wander
            sp = np.sort(sp)[::-1]
            motion[i] = float(np.mean(sp[: max(1, len(sp) // 2)]))
    return {"t": t, "ball_speed": ball_speed, "ball_seen": ball_seen, "motion": motion}


def detect_whistles(video_path: str, start: float = 0.0, end: Optional[float] = None) -> List[float]:
    """Return timestamps (s) of whistle-like tones. Empty list if no audio / ffmpeg missing."""
    sr = 16000
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout
    except Exception:
        return []
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if len(a) < sr:
        return []
    win, hop = 800, 400  # 50 ms / 25 ms
    nfr = (len(a) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nfr)[:, None]
    spec = np.abs(np.fft.rfft(a[idx] * np.hanning(win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band = (freqs >= 2000) & (freqs <= 4500)
    tot = spec.sum(1) + 1e-10
    be = spec[:, band]
    ratio = be.sum(1) / tot
    tonal = be.max(1) / (be.mean(1) + 1e-10)  # whistle = narrow peak
    loud = tot > np.percentile(tot, 60)
    hit = (ratio > 0.45) & (tonal > 8) & loud
    hit = _smooth(hit.astype(float), 5) > 0.6
    times, i = [], 0
    while i < nfr:
        if hit[i]:
            j = i
            while j < nfr and hit[j]:
                j += 1
            if (j - i) * hop / sr >= 0.15:
                times.append(round(start + (i * hop + win / 2) / sr, 2))
            i = j
        else:
            i += 1
    return times


def segment_rallies(frames: List[Dict], diag: float, whistles: Optional[List[float]] = None,
                    high: float = 0.45, low: float = 0.28, min_len: float = 2.5, merge_gap: float = 1.5,
                    pad_pre: float = 1.0, pad_post: float = 1.0) -> Dict:
    if len(frames) < 3:
        return {"rallies": [], "score": []}
    s = activity_signals(frames, diag)
    t = s["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 15
    w = max(1, int(round(1.0 / dt)))  # 1-second windows
    ball_act = _smooth(((s["ball_speed"] > 0.15) & (s["ball_seen"] > 0)).astype(float), w)
    ball_vis = _smooth(s["ball_seen"], w)
    mot = _smooth(s["motion"], w)
    ref = np.percentile(mot[mot > 0], 85) if (mot > 0).any() else 1.0
    mot_n = np.clip(mot / max(ref, 0.25), 0, 1)
    score = 0.45 * np.clip(ball_act * 2.0, 0, 1) + 0.15 * ball_vis + 0.40 * mot_n
    score = _smooth(score, w)

    # adapt thresholds to this clip: split the score histogram (Otsu) between dead and live modes
    hist, edges = np.histogram(score, bins=50, range=(0, 1))
    if hist.sum() > 20:
        c = (edges[:-1] + edges[1:]) / 2
        best, thr = -1.0, None
        for k in range(5, 45):
            w0, w1 = hist[:k].sum(), hist[k:].sum()
            if w0 == 0 or w1 == 0:
                continue
            m0, m1 = (hist[:k] * c[:k]).sum() / w0, (hist[k:] * c[k:]).sum() / w1
            var = w0 * w1 * (m1 - m0) ** 2
            if var > best:
                best, thr, gap = var, c[k], m1 - m0
        if thr is not None and gap > 0.2:  # clearly bimodal
            high, low = max(high, thr + 0.06), max(low, thr - 0.04)
    # hysteresis thresholding
    live, on = np.zeros(len(score), bool), False
    for i, v in enumerate(score):
        on = v >= high if not on else v >= low
        live[i] = on
    # a whistle always marks a boundary (serve authorisation or end of rally)
    for wt in whistles or []:
        live[(t >= wt - 0.1) & (t <= wt + 0.4)] = False
    segs, i = [], 0
    while i < len(live):
        if live[i]:
            j = i
            while j < len(live) and live[j]:
                j += 1
            segs.append([t[i], t[j - 1]])
            i = j
        else:
            i += 1
    merged: List[List[float]] = []
    for a, b in segs:
        crosses = any(merged and merged[-1][1] <= wt <= a for wt in (whistles or []))
        if merged and a - merged[-1][1] < merge_gap and not crosses:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rallies = []
    t0, t1 = float(t[0]), float(t[-1])
    for a, b in merged:
        if b - a < min_len:
            continue
        cues = []
        clip_a, clip_b = a - pad_pre, b + pad_post
        if whistles:
            before = [x for x in whistles if a - 4.0 <= x <= a + 0.5]
            after = [x for x in whistles if b - 2.0 <= x <= b + 2.5]
            if before:  # serve whistle: include it in the clip, but play starts at the serve
                clip_a = min(clip_a, before[-1] - 0.3); cues.append("start_whistle")
            if after:   # end whistle: play is over by then
                b = min(b, after[0])
                clip_b = max(b + pad_post, after[0] + 0.5); cues.append("end_whistle")
        sel = (t >= a) & (t <= b)
        rallies.append({
            "start": round(max(t0, clip_a), 2), "end": round(min(t1, clip_b), 2),
            "core_start": round(float(a), 2), "core_end": round(float(b), 2),
            "confidence": round(float(np.mean(score[sel])) if sel.any() else 0.0, 3), "cues": cues,
        })
    for k, r in enumerate(rallies, 1):
        r["id"] = k
        if k > 1 and r["start"] < rallies[k - 2]["end"]:  # padding must not overlap
            mid = round((rallies[k - 2]["core_end"] + r["core_start"]) / 2, 2)
            rallies[k - 2]["end"], r["start"] = mid, mid
    return {"rallies": rallies, "score": [round(float(v), 3) for v in score]}
