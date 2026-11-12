"""End-to-end analysis of one video."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import __version__
from .actions import spot_actions
from .ball import BallTracker, clean_trajectory
from .camera import CameraMotion
from .court import CourtModel, detect_court, project, zone_of
from .detect import Detector, filter_players
from .export import cut_clips, write_csvs, write_json, zone_stats
from .rally import detect_whistles, segment_rallies
from .render import draw_frame
from .video import is_scene_cut, iter_frames, probe


@dataclass
class Config:
    model: str = "yolo11n-seg.pt"   # players (and ball, when ball_model == "yolo")
    ball_model: str = "auto"        # auto | yolo | path to a TrackNetV3 checkpoint
    ball_thresh: float = 0.5        # TrackNet heatmap threshold (lower = more detections)
    imgsz: int = 640
    stride: int = 2               # analyse every Nth frame (2 => 15 fps for 30 fps video)
    max_width: int = 960
    conf: float = 0.3
    ball_conf: float = 0.15
    masks: bool = True
    motion_ball: bool = True
    court: str = "auto"          # auto | off | path/to/calibration.json
    court_retry_s: float = 1.0
    whistle: bool = True
    rally_high: float = 0.45
    rally_low: float = 0.28
    min_rally: float = 2.5
    start: float = 0.0
    end: Optional[float] = None
    clips: bool = True
    overlay: bool = True
    actions: bool = True


TRACKNET_NAMES = ("TrackNet_best.pt", "tracknet_best.pt", os.path.join("weights", "TrackNet_best.pt"),
                  os.path.join("weights", "tracknet_best.pt"), os.path.join("ckpts", "TrackNet_best.pt"))


def resolve_ball_model(choice: str) -> Optional[str]:
    """Return a TrackNet checkpoint path, or None to use YOLO + motion blobs."""
    if choice == "yolo":
        return None
    if choice != "auto":
        if not os.path.isfile(choice):
            raise FileNotFoundError(f"TrackNet weights not found: {choice}")
        return choice
    for p in TRACKNET_NAMES:
        if os.path.isfile(p):
            return p
    return None


def _log(msg: str, quiet: bool):
    if not quiet:
        print(msg, flush=True)


def analyze(video: str, out_dir: str, cfg: Config = Config(), quiet: bool = False) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    info = probe(video)
    _log(f"[vballseg] {os.path.basename(video)}: {info.width}x{info.height} @ {info.fps:.1f} fps, "
         f"{info.duration:.1f}s", quiet)
    tn_path = resolve_ball_model(cfg.ball_model)
    det = Detector(cfg.model, cfg.imgsz, cfg.conf, cfg.ball_conf, cfg.masks, detect_ball=tn_path is None)
    tn = None
    if tn_path:
        from .tracknet import TrackNetBall, shot_backgrounds
        _log(f"[vballseg] ball: TrackNetV3 ({tn_path}); building background images...", quiet)
        tn = TrackNetBall(tn_path, shot_backgrounds(video, cfg.start, cfg.end), threshold=cfg.ball_thresh)
    else:
        _log("[vballseg] ball: YOLO 'sports ball' + motion tracker (no TrackNet weights found)", quiet)
    cam = CameraMotion()
    ball: Optional[BallTracker] = None
    manual: Optional[CourtModel] = None
    court: Optional[CourtModel] = None
    frames: List[Dict] = []
    prev_gray, last_try, diag, scale0 = None, -1e9, 1000.0, 1.0
    shot, cuts = 0, []
    t_start = time.time()
    fps_eff = info.fps / cfg.stride

    for fidx, t, frame, scale in iter_frames(video, cfg.stride, cfg.max_width, cfg.start, cfg.end):
        h, w = frame.shape[:2]
        if ball is None:
            ball = BallTracker(frame.shape, fps_eff, cfg.motion_ball)
            diag, scale0 = float(np.hypot(w, h)), scale
            if cfg.court not in ("auto", "off"):
                manual = CourtModel.load(cfg.court, scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- shot boundaries (mixed / edited clips)
        if prev_gray is not None and is_scene_cut(prev_gray, gray):
            shot += 1
            cuts.append(round(t, 2))
            if tn is not None:
                tn.flush()  # TrackNet windows must not span a cut
            det.reset_tracks(); cam.reset(); ball.reset()
            court = None if cfg.court == "auto" else court
            last_try = -1e9
        prev_gray = gray

        # --- court model for this shot
        if manual is not None and court is None and t >= manual.ref_time - 1e-3:
            court = manual
            cam.reset()
        elif cfg.court == "auto":
            retry = cfg.court_retry_s if court is None else 10.0
            if t - last_try >= retry:
                last_try = t
                cand = detect_court(frame, t)
                if cand is not None and cand.confidence >= 0.5 and \
                        (court is None or cand.confidence >= court.confidence):
                    court = cand
                    cam.reset()

        d = det(frame)
        H_cur2ref = cam.update(gray, d["players"])
        H_img2court = None
        if court is not None and not cam.ok:
            # lost camera-motion lock: chained transform is no longer trustworthy
            if court.source == "auto":
                court, last_try = None, -1e9
            else:
                cam.reset(); cam.update(gray, d["players"])  # assume camera held still
        if court is not None and cam.ok:
            H_img2court = court.H_img2court @ H_cur2ref

        def side_fn(p):
            if H_img2court is None:
                return None
            X, Y = project(H_img2court, [p["foot"]])[0]
            return zone_of(X, Y)["side"]

        players = filter_players(d["players"], h, side_fn if H_img2court is not None else None)
        if H_img2court is not None:
            feet = project(H_img2court, [p["foot"] for p in players]) if players else []
            for p, (X, Y) in zip(players, feet):
                p["court"] = [round(float(X), 2), round(float(Y), 2)]
                p.update(zone_of(X, Y))
        b = None if tn is not None else ball.update(gray, d["ball_dets"], players, cam.last_H)
        rec = {"frame": fidx, "t": round(t, 3), "shot": shot, "players": players, "ball": b}
        if tn is not None:
            tn.push(frame, t, lambda bb, r=rec: r.__setitem__("ball", bb))
        if H_img2court is not None:
            rec["H_court2img"] = np.linalg.inv(H_img2court).round(6).ravel().tolist()
        frames.append(rec)
        if not quiet and len(frames) % 100 == 0:
            el = time.time() - t_start
            _log(f"  t={t:7.1f}s  frames={len(frames)}  {len(frames) / el:.1f} fps  "
                 f"court={'yes' if H_img2court is not None else 'no'}", quiet)

    if tn is not None:
        tn.flush()
    if not frames:
        raise RuntimeError("No frames analysed")
    # --- post-processing
    balls = clean_trajectory([f["ball"] for f in frames], max_gap=max(2, int(0.4 * fps_eff)), diag=diag)
    for f, b in zip(frames, balls):
        f["ball"] = b
    whistles = detect_whistles(video, cfg.start, cfg.end) if cfg.whistle else []
    rs = segment_rallies(frames, diag, whistles, cfg.rally_high, cfg.rally_low, cfg.min_rally)
    rallies = rs["rallies"]
    events = spot_actions(frames, rallies, diag) if cfg.actions else []
    stats = zone_stats(frames, rallies, events)
    court_frac = float(np.mean([("H_court2img" in f) for f in frames]))
    elapsed = time.time() - t_start

    result = {
        "version": __version__, "video": os.path.abspath(video), "config": asdict(cfg),
        "video_info": {"fps": info.fps, "width": info.width, "height": info.height, "duration": info.duration},
        "analysis": {"ball_model": tn_path or "yolo", "scale": scale0, "frames": len(frames), "seconds": round(elapsed, 1),
                     "court_coverage": round(court_frac, 3), "scene_cuts": cuts, "whistles": whistles},
        "rallies": rallies, "events": events, "stats": stats,
    }
    write_json(os.path.join(out_dir, "summary.json"), result)
    write_json(os.path.join(out_dir, "frames.json"), {"scale": scale0, "frames": frames, "rally_score": rs["score"]})
    write_csvs(out_dir, rallies, events)
    _log(f"[vballseg] {len(rallies)} rallies, {len(events)} actions, court mapped on "
         f"{100 * court_frac:.0f}% of frames ({elapsed:.0f}s)", quiet)

    if cfg.clips and rallies:
        cut_clips(video, rallies, out_dir)
        _log(f"[vballseg] clips -> {os.path.join(out_dir, 'clips')}", quiet)
    if cfg.overlay:
        render_overlay(video, frames, rallies, events, os.path.join(out_dir, "overlay.mp4"), cfg, info.fps)
        _log(f"[vballseg] overlay -> {os.path.join(out_dir, 'overlay.mp4')}", quiet)
    return result


def render_overlay(video: str, frames: List[Dict], rallies, events, path: str, cfg: Config, fps: float):
    by_frame = {f["frame"]: i for i, f in enumerate(frames)}
    tmp = path + ".tmp.mp4"
    writer = None
    for fidx, t, img, _ in iter_frames(video, cfg.stride, cfg.max_width, cfg.start, cfg.end):
        i = by_frame.get(fidx)
        if i is None:
            continue
        rec = frames[i]
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps / cfg.stride, (w, h))
        trail = [frames[j]["ball"]["xy"] if frames[j]["ball"] else None for j in range(max(0, i - 8), i + 1)]
        rally = next((r for r in rallies if r["start"] <= t <= r["end"]), None)
        evs = [e for e in events if 0 <= t - e["t"] <= 0.8]
        writer.write(draw_frame(img, rec, trail, rally, evs))
    if writer is not None:
        writer.release()
        # re-encode to H.264 so it plays everywhere
        rc = os.system(f'ffmpeg -y -v error -i "{tmp}" -c:v libx264 -preset veryfast -crf 23 '
                       f'-pix_fmt yuv420p -movflags +faststart "{path}"')
        if rc == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, path)
