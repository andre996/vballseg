"""Fine-tune (or train) TrackNetV3 for volleyball on your own labelled clips. CPU works; a GPU is faster.

Labels: one CSV per video in TrackNet format  ->  Frame,Visibility,X,Y   (original-video pixels;
Visibility 0 = ball not visible). Make them with tools/label_ball.py.

    python tools/train_tracknet.py --data clip1.mp4:clip1.csv clip2.mp4:clip2.csv \
        --init TrackNet_best.pt --out TrackNet_volley.pt --epochs 10 --frame-step 2

--frame-step should match the --stride you analyse with (TrackNet learns the ball's motion between
the frames it is shown). Then:  vballseg analyze match.mp4 --ball-model TrackNet_volley.pt
"""
import argparse
import csv
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vballseg.tracknet import TN_H, TN_W, TrackNet, load_tracknet  # noqa: E402

SIGMA = 2.5


def load_clip(video, labels, step):
    lab = {int(r["Frame"]): r for r in csv.DictReader(open(labels))}
    cap = cv2.VideoCapture(video)
    frames, pts, known, i = [], [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            frames.append(cv2.resize(f, (TN_W, TN_H), interpolation=cv2.INTER_AREA)[..., ::-1].copy())
            r = lab.get(i)
            known.append(r is not None)
            if r and int(float(r["Visibility"])) > 0:
                pts.append((float(r["X"]) * TN_W / w, float(r["Y"]) * TN_H / h))
            else:
                pts.append(None)
        i += 1
    cap.release()
    # background: median of up to 100 evenly spaced frames
    sel = frames[:: max(1, len(frames) // 100)]
    median = np.median(np.stack(sel), 0).astype(np.uint8)
    return np.stack(frames), pts, median, known


def heatmap(p):
    if p is None:
        return np.zeros((TN_H, TN_W), np.float32)
    yy, xx = np.mgrid[0:TN_H, 0:TN_W]
    g = np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * SIGMA ** 2))
    return (g > 0.5).astype(np.float32) * g  # TrackNet uses a truncated gaussian blob


def make_input(frames, median, bg_mode):
    chans = []
    med = median.astype(np.float32)
    if bg_mode == "concat":
        chans.append(np.moveaxis(med, -1, 0))
    for f in frames.astype(np.float32):
        if bg_mode == "subtract":
            chans.append(np.abs(f - med).sum(2, keepdims=True).clip(0, 255).transpose(2, 0, 1))
        elif bg_mode == "subtract_concat":
            chans += [np.moveaxis(f, -1, 0), np.abs(f - med).sum(2, keepdims=True).clip(0, 255).transpose(2, 0, 1)]
        else:
            chans.append(np.moveaxis(f, -1, 0))
    return np.concatenate(chans, 0) / 255.0


def wbce(y_pred, y):
    """TrackNetV2/V3 weighted binary cross-entropy (focal-style)."""
    eps = 1e-7
    p = y_pred.clamp(eps, 1 - eps)
    return -((1 - p) ** 2 * y * torch.log(p) + p ** 2 * (1 - y) * torch.log(1 - p)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="video.mp4:labels.csv pairs")
    ap.add_argument("--init", help="checkpoint to start from (recommended: upstream TrackNet_best.pt)")
    ap.add_argument("--out", default="TrackNet_volley.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = no limit)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.init:
        net, L, bg = load_tracknet(a.init, dev)
        net.train()
    else:
        L, bg = a.seq_len, "concat"
        net = TrackNet((L + 1) * 3, L).to(dev)

    clips = []
    for pair in a.data:
        v, c = pair.rsplit(":", 1)
        fr, pts, med, known = load_clip(v, c, a.frame_step)
        clips.append((fr, pts, med, known))
        print(f"{v}: {len(fr)} frames used, {sum(known)} labelled, {sum(p is not None for p in pts)} with ball")
    # windows that contain at least one visible ball (plus some empty ones so it learns "no ball")
    win = []
    for ci, (fr, pts, _, known) in enumerate(clips):
        for s in range(0, len(fr) - L + 1, max(1, L // 2)):
            if not all(known[s:s + L]):
                continue  # window touches unlabelled frames (outside your labelled sections)
            if any(p is not None for p in pts[s:s + L]) or random.random() < 0.15:
                win.append((ci, s))
    print(f"{len(win)} training windows, seq_len={L}, bg_mode={bg}, device={dev}")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(win)
        tot = 0.0
        for b in range(0, len(win), a.batch):
            xs, ys = [], []
            for ci, s in win[b:b + a.batch]:
                fr, pts, med, _ = clips[ci]
                f = fr[s:s + L]
                if random.random() < 0.5:  # horizontal flip augmentation
                    f, med_ = f[:, :, ::-1], med[:, ::-1]
                    p_ = [None if p is None else (TN_W - 1 - p[0], p[1]) for p in pts[s:s + L]]
                else:
                    med_, p_ = med, pts[s:s + L]
                xs.append(make_input(np.ascontiguousarray(f), np.ascontiguousarray(med_), bg))
                ys.append(np.stack([heatmap(p) for p in p_]))
            x = torch.from_numpy(np.stack(xs)).float().to(dev)
            y = torch.from_numpy(np.stack(ys)).float().to(dev)
            loss = wbce(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            step += 1
            if step % 10 == 0:
                print(f"epoch {ep + 1} step {step} loss {loss.item():.5f} ({time.time() - t0:.0f}s)", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        torch.save({"model": net.state_dict(), "param_dict": {"seq_len": L, "bg_mode": bg}}, a.out)
        print(f"epoch {ep + 1}: mean loss {tot / max(1, (len(win) + a.batch - 1) // a.batch):.5f} -> saved {a.out}")
        if a.max_steps and step >= a.max_steps:
            break


if __name__ == "__main__":
    main()
