"""Command-line interface.

  vballseg analyze match.mp4 -o out/            full pipeline
  vballseg analyze clips/*.mp4 -o out/          batch (one sub-folder per video)
  vballseg calibrate match.mp4 -o court.json    click court points (needs a display)
  vballseg calibrate match.mp4 --points "x,y x,y x,y x,y" -o court.json   (headless)
  vballseg clips out/summary.json --pad 2       re-cut clips with different padding
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

from .court import POINT_PRESETS, CourtModel, draw_court


def _secs(v) -> float:
    """'180', '3:00', '1:02:03' -> seconds"""
    sec = 0.0
    for part in str(v).split(":"):
        sec = sec * 60 + float(part)
    return sec


def cmd_analyze(a):
    from .pipeline import Config, analyze  # heavy imports only when needed

    vids = []
    for pat in a.videos:
        vids += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    if os.path.isdir(vids[0]) if vids else False:
        d = vids[0]
        vids = sorted(glob.glob(os.path.join(d, "*.mp4")) + glob.glob(os.path.join(d, "*.mov"))
                      + glob.glob(os.path.join(d, "*.MOV")) + glob.glob(os.path.join(d, "*.MP4")))
    if not vids:
        sys.exit("no input videos")
    cfg = Config(model=a.model, ball_model=a.ball_model, ball_thresh=a.ball_thresh, imgsz=a.imgsz, stride=a.stride, max_width=a.width, conf=a.conf,
                 masks=not a.no_masks, motion_ball=not a.no_motion_ball, court=a.court,
                 whistle=not a.no_whistle, rally_high=a.rally_high, rally_low=a.rally_low,
                 min_rally=a.min_rally, start=a.start, end=a.end, clips=not a.no_clips,
                 overlay=not a.no_overlay, actions=not a.no_actions)
    import dataclasses
    from .video import probe
    for v in vids:
        out = a.out if len(vids) == 1 else os.path.join(a.out, os.path.splitext(os.path.basename(v))[0])
        try:
            vcfg = cfg
            if a.last:
                dur = probe(v).duration
                if dur <= 0:
                    raise ValueError("cannot read video length; use --start instead")
                st = max(0.0, dur - _secs(a.last))
                vcfg = dataclasses.replace(cfg, start=st, end=None)
                if not a.quiet:
                    print(f"[vballseg] --last {a.last}: analysing {st:.0f}s-{dur:.0f}s of {os.path.basename(v)}")
            analyze(v, out, vcfg, quiet=a.quiet)
        except Exception as e:  # keep going in batch mode
            print(f"[vballseg] FAILED {v}: {e}", file=sys.stderr)
            if len(vids) == 1:
                raise


def _grab(video, t):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read frame at {t}s")
    return frame


def cmd_calibrate(a):
    frame = _grab(a.video, a.time)
    court_pts = POINT_PRESETS[a.preset]
    names = {"corners": ["near-left corner", "near-right corner", "far-right corner", "far-left corner"],
             "far_half": ["centre line, left end", "centre line, right end", "far-right corner", "far-left corner"],
             "near_half": ["near-left corner", "near-right corner", "centre line, right end",
                           "centre line, left end"]}[a.preset]
    if a.points:
        img_pts = [[float(v) for v in p.split(",")] for p in a.points.split()]
    else:
        img_pts, disp = [], frame.copy()
        s = min(1.0, 1400 / frame.shape[1])
        win = "vballseg calibrate - click points, Backspace = undo, Enter = done"

        def on_click(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(img_pts) < 4:
                img_pts.append([x / s, y / s])

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_click)
        while True:
            disp = cv2.resize(frame, None, fx=s, fy=s)
            for i, (x, y) in enumerate(img_pts):
                cv2.circle(disp, (int(x * s), int(y * s)), 6, (0, 0, 255), -1)
                cv2.putText(disp, str(i + 1), (int(x * s) + 8, int(y * s)), 0, 0.7, (0, 0, 255), 2)
            msg = f"Click {len(img_pts) + 1}/4: {names[len(img_pts)]}" if len(img_pts) < 4 else "Enter to save"
            cv2.putText(disp, msg, (10, 30), 0, 0.8, (0, 0, 0), 4)
            cv2.putText(disp, msg, (10, 30), 0, 0.8, (0, 255, 255), 2)
            if len(img_pts) == 4:
                cm = CourtModel.from_points(img_pts, court_pts)
                draw_court(disp, np.diag([s, s, 1]) @ cm.H_court2img)
            cv2.imshow(win, disp)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10) and len(img_pts) == 4:
                break
            if k == 8 and img_pts:
                img_pts.pop()
            if k == 27:
                sys.exit("cancelled")
        cv2.destroyAllWindows()
    if len(img_pts) != 4:
        sys.exit("need exactly 4 points")
    cm = CourtModel.from_points(img_pts, court_pts, "manual", a.time)
    json.dump(cm.to_dict(), open(a.out, "w"), indent=1)
    preview = frame.copy()
    draw_court(preview, cm.H_court2img)
    cv2.imwrite(os.path.splitext(a.out)[0] + "_preview.jpg", preview)
    print(f"saved {a.out} (+ preview image). Use: vballseg analyze {a.video} --court {a.out}")


def cmd_clips(a):
    from .export import cut_clips

    s = json.load(open(a.summary))
    rallies = []
    for r in s["rallies"]:
        r = dict(r)
        r["start"] = max(0.0, r["core_start"] - a.pad)
        r["end"] = r["core_end"] + a.pad
        if r["end"] - r["start"] >= a.min_len:
            rallies.append(r)
    out = a.out or os.path.dirname(os.path.abspath(a.summary))
    cut_clips(s["video"], rallies, out, reel=True)
    print(f"{len(rallies)} clips -> {os.path.join(out, 'clips')}")


def main(argv=None):
    ap = argparse.ArgumentParser("vballseg", description="Volleyball video segmentation (CPU)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="run the full pipeline")
    p.add_argument("videos", nargs="+", help="video file(s), glob, or a folder")
    p.add_argument("-o", "--out", default="vballseg_out")
    p.add_argument("--model", default="yolo11n-seg.pt", help="YOLO seg/detect weights for players")
    p.add_argument("--ball-model", default="auto",
                   help="'auto' (TrackNet if TrackNet_best.pt is found, else YOLO), 'yolo', or a TrackNetV3 .pt path")
    p.add_argument("--ball-thresh", type=float, default=0.5,
                   help="TrackNet heatmap threshold; lower it (e.g. 0.3) if the ball is often missed")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=2, help="analyse every Nth frame")
    p.add_argument("--width", type=int, default=960, help="resize frames to this max width")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--court", default="auto", help="'auto', 'off', or a calibration JSON")
    p.add_argument("--start", type=_secs, default=0.0, help="start time, seconds or m:ss")
    p.add_argument("--end", type=_secs, default=None, help="end time, seconds or m:ss")
    p.add_argument("--last", help="analyse only the last part of each video, e.g. 3:00 or 180")
    p.add_argument("--rally-high", type=float, default=0.45, help="activity score to start a rally")
    p.add_argument("--rally-low", type=float, default=0.28, help="activity score to end a rally")
    p.add_argument("--min-rally", type=float, default=2.5, help="minimum rally length (s)")
    for flag in ("masks", "motion-ball", "whistle", "clips", "overlay", "actions"):
        p.add_argument(f"--no-{flag}", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_analyze)

    c = sub.add_parser("calibrate", help="mark court points for homography")
    c.add_argument("video")
    c.add_argument("-o", "--out", default="court.json")
    c.add_argument("--time", type=float, default=0.0, help="timestamp of frame to calibrate on")
    c.add_argument("--preset", choices=list(POINT_PRESETS), default="corners")
    c.add_argument("--points", help='headless: "x1,y1 x2,y2 x3,y3 x4,y4" in original-video pixels')
    c.set_defaults(fn=cmd_calibrate)

    k = sub.add_parser("clips", help="re-cut rally clips from a summary.json")
    k.add_argument("summary")
    k.add_argument("--pad", type=float, default=1.0)
    k.add_argument("--min-len", type=float, default=2.0)
    k.add_argument("-o", "--out")
    k.set_defaults(fn=cmd_clips)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
