"""Generate a synthetic volleyball clip with known ground truth for smoke-testing the pipeline.

Real people (cut out of Ultralytics' sample image) are composited onto a perspective court,
with scripted rallies (serve -> receive -> set -> attack -> dig -> set -> attack), dead time,
handheld camera shake/pan, and referee whistles in the audio track.

    python tools/make_synthetic.py out_dir/
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
SR = 16000
# court corners in image (near-left, near-right, far-right, far-left): elevated end view
IMG_CORNERS = np.float32([[170, 680], [1110, 680], [850, 250], [430, 250]])
COURT = np.float32([[0, 0], [9, 0], [9, 18], [0, 18]])
Hc2i, _ = cv2.findHomography(COURT, IMG_CORNERS)


def c2i(X, Y):
    p = cv2.perspectiveTransform(np.float64([[[X, Y]]]), Hc2i)[0, 0]
    return p


def px_per_m(X, Y):
    a, b = c2i(X - 0.5, Y), c2i(X + 0.5, Y)
    return float(np.linalg.norm(b - a))


def sprites():
    from ultralytics import YOLO
    import ultralytics
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    r = YOLO("yolo11n-seg.pt")(img, classes=[0], verbose=False, retina_masks=True)[0]
    out = []
    for box, m in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.masks.data.cpu().numpy()):
        x1, y1, x2, y2 = box
        if y2 - y1 < 300:
            continue
        a = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
        out.append(np.dstack([img[y1:y2, x1:x2], a]))
    return out


def paste(frame, spr, foot_xy, height_px):
    s = height_px / spr.shape[0]
    sp = cv2.resize(spr, (max(2, int(spr.shape[1] * s)), max(2, int(height_px))), interpolation=cv2.INTER_AREA)
    x0, y0 = int(foot_xy[0] - sp.shape[1] / 2), int(foot_xy[1] - sp.shape[0])
    h, w = sp.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = sp[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    al = sub[..., 3:4] / 255.0
    frame[fy0:fy1, fx0:fx1] = (al * sub[..., :3] + (1 - al) * frame[fy0:fy1, fx0:fx1]).astype(np.uint8)


def background():
    bg = np.full((H, W, 3), (60, 90, 150), np.uint8)  # wall / stands
    bg[230:] = (70, 120, 185)                          # floor
    court = np.zeros_like(bg)
    cv2.fillPoly(court, [np.int32(IMG_CORNERS)], (60, 100, 200))
    m = court.any(2)
    bg[m] = court[m]
    rng = np.random.default_rng(0)
    bg = np.clip(bg + rng.normal(0, 4, bg.shape), 0, 255).astype(np.uint8)
    for a, b in [((0, 0), (9, 0)), ((9, 0), (9, 18)), ((9, 18), (0, 18)), ((0, 18), (0, 0)),
                 ((0, 9), (9, 9)), ((0, 6), (9, 6)), ((0, 12), (9, 12))]:
        cv2.line(bg, tuple(np.int32(c2i(*a))), tuple(np.int32(c2i(*b))), (245, 245, 245), 3, cv2.LINE_AA)
    # texture on stands so camera-motion tracking has features
    for _ in range(400):
        x, y = rng.integers(0, W), rng.integers(0, 225)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(2, 6)), tuple(int(v) for v in rng.integers(30, 220, 3)), -1)
    return bg


HOME_A = [(7.5, 1.5), (7.5, 7.5), (4.5, 7.5), (1.5, 7.5), (1.5, 1.5), (4.5, 3)]
HOME_B = [(1.5, 16.5), (1.5, 10.5), (4.5, 10.5), (7.5, 10.5), (7.5, 16.5), (4.5, 15)]

# (time offset s, side, player index, contact court XY, type)
RALLY_SCRIPT = [
    (0.0, "A", 0, (7.5, -1.0), "serve"),
    (1.5, "B", 5, (4.5, 14.5), "receive"),
    (2.6, "B", 2, (5.0, 10.0), "set"),
    (3.6, "B", 3, (7.5, 10.3), "attack"),
    (4.3, "A", 4, (2.5, 3.0), "dig"),
    (5.4, "A", 2, (4.5, 8.0), "set"),
    (6.4, "A", 3, (1.5, 8.2), "attack"),
    (7.1, None, None, (6.0, 15.0), "land"),
]
RALLY_STARTS = [4.0, 18.0, 32.0]
DURATION = 44.0


def main(out):
    os.makedirs(out, exist_ok=True)
    spr = sprites()
    bg = background()
    tmp = os.path.join(out, "_v.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(1)
    pos = {"A": [np.array(p, float) for p in HOME_A], "B": [np.array(p, float) for p in HOME_B]}
    truth = {"rallies": [], "events": []}
    for rs in RALLY_STARTS:
        truth["rallies"].append({"start": rs, "end": rs + RALLY_SCRIPT[-1][0]})
        for dt, side, pi, xy, typ in RALLY_SCRIPT[:-1]:
            truth["events"].append({"t": rs + dt, "type": typ, "side": side})
    n = int(DURATION * FPS)
    labels = []  # TrackNet CSV format: Frame,Visibility,X,Y (original pixels)
    for i in range(n):
        t = i / FPS
        rally = next((rs for rs in RALLY_STARTS if rs <= t <= rs + RALLY_SCRIPT[-1][0] + 0.3), None)
        targets = {"A": [np.array(p, float) for p in HOME_A], "B": [np.array(p, float) for p in HOME_B]}
        ball = None  # (X, Y, Z)
        if rally is not None:
            rt = t - rally
            # contact players move to their contact spot ahead of time
            for k, (dt, side, pi, xy, typ) in enumerate(RALLY_SCRIPT[:-1]):
                if dt - 1.2 <= rt <= dt + 0.2:
                    targets[side][pi] = np.array(xy)
            for k in range(len(RALLY_SCRIPT) - 1):
                t0, t1 = RALLY_SCRIPT[k][0], RALLY_SCRIPT[k + 1][0]
                if t0 <= rt < t1:
                    p0, p1 = np.array(RALLY_SCRIPT[k][3]), np.array(RALLY_SCRIPT[k + 1][3])
                    f = (rt - t0) / (t1 - t0)
                    z0 = 2.4
                    z1 = 0.1 if RALLY_SCRIPT[k + 1][4] == "land" else 2.4
                    apex = {"set": 4.5, "serve": 4.0, "attack": 2.8}.get(RALLY_SCRIPT[k][4], 5.0)
                    if RALLY_SCRIPT[k][4] == "attack":
                        z = z0 + (z1 - z0) * f + 0.6 * np.sin(np.pi * f)
                    else:
                        z = z0 + (z1 - z0) * f + (apex - z0) * 4 * f * (1 - f)
                    xy = p0 + f * (p1 - p0)
                    ball = (xy[0], xy[1], z)
            speed = 4.0
        else:
            speed = 0.6
            # dead time: players wander slightly
            for s in "AB":
                for k in range(6):
                    targets[s][k] = targets[s][k] + 0.4 * np.array([np.sin(0.3 * t + k), np.cos(0.25 * t + 2 * k)])
        frame = bg.copy()
        people = []
        for s in "AB":
            for k in range(6):
                d = targets[s][k] - pos[s][k]
                step = min(np.linalg.norm(d), speed / FPS)
                if np.linalg.norm(d) > 1e-6:
                    pos[s][k] = pos[s][k] + d / np.linalg.norm(d) * step
                if rally is not None:
                    pos[s][k] = pos[s][k] + rng.normal(0, 0.01, 2)
                people.append((s, k, pos[s][k].copy()))
        people.sort(key=lambda q: -q[2][1])  # far players first
        for s, k, (X, Y) in people:
            ppm = px_per_m(X, Y)
            paste(frame, spr[(k + (0 if s == "A" else 2)) % len(spr)], c2i(X, Y), 1.85 * ppm)
        bxy = None
        if ball is not None:
            X, Y, Z = ball
            g = c2i(X, Y)
            ppm = px_per_m(X, Y)
            bxy = (int(g[0]), int(g[1] - Z * ppm))
            cv2.circle(frame, bxy, max(3, int(0.21 * ppm / 2) + 2), (60, 230, 250), -1, cv2.LINE_AA)
        # handheld camera: shake + slow pan between 20 s and 28 s
        pan = 60 * np.clip((t - 20) / 8, 0, 1)
        dx = 3 * np.sin(2.1 * t) + pan
        dy = 2 * np.sin(1.7 * t + 1)
        M = np.float32([[1, 0, -dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        vw.write(frame)
        if bxy is not None and 0 <= bxy[0] - dx < W and 0 <= bxy[1] + dy < H:
            labels.append(f"{i},1,{bxy[0] - dx:.1f},{bxy[1] + dy:.1f}")
        else:
            labels.append(f"{i},0,0,0")
    vw.release()
    # audio: crowd noise + whistles before serve and after rally end
    a = np.random.default_rng(2).normal(0, 0.03, int(DURATION * SR)).astype(np.float32)
    for rs in RALLY_STARTS:
        for wt in (rs - 1.5, rs + RALLY_SCRIPT[-1][0] + 0.4):
            i0 = int(wt * SR)
            tt = np.arange(int(0.5 * SR)) / SR
            a[i0:i0 + len(tt)] += 0.4 * np.sin(2 * np.pi * 3200 * tt) * (1 + 0.2 * np.sin(2 * np.pi * 30 * tt))
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()
    wav = os.path.join(out, "_a.raw")
    open(wav, "wb").write(pcm)
    dst = os.path.join(out, "synthetic_match.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", wav,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", dst], check=True)
    os.remove(tmp); os.remove(wav)
    json.dump(truth, open(os.path.join(out, "ground_truth.json"), "w"), indent=1)
    open(os.path.join(out, "ball_labels.csv"), "w").write("Frame,Visibility,X,Y\n" + "\n".join(labels) + "\n")
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
