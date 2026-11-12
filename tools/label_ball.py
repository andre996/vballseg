"""Click-to-label ball positions for TrackNet fine-tuning (needs a display).

    python tools/label_ball.py clip.mp4                          -> writes clip.csv (Frame,Visibility,X,Y)
    python tools/label_ball.py match.mp4 --start 1:25 --end 1:40   label only that section
    python tools/label_ball.py match.mp4 --start 300 --end 315     (seconds also work)

Run it again with other --start/--end values to add more sections to the same CSV.
Only frames you actually label are saved, so frames outside your sections are ignored in training.

Controls
  left click     ball here (auto-advances)
  n / space      no ball visible in this frame (auto-advances)
  d / a          next / previous frame without changing the label
  f / b          jump +10 / -10 frames
  s              save            q / Esc   save and quit

Tip: label every frame of a few short rallies (10-20 s each) from the camera angles you
actually film; 1,500-3,000 labelled frames is a solid start. Existing CSVs are resumed.
"""
import argparse
import csv
import os

import cv2


def parse_time(v):
    """'90', '90.5', '1:30' or '1:02:03' -> seconds"""
    if v is None:
        return None
    sec = 0.0
    for part in str(v).split(":"):
        sec = sec * 60 + float(part)
    return sec


def main(video, out=None, start=None, end=None):
    out = out or os.path.splitext(video)[0] + ".csv"
    cap = cv2.VideoCapture(video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0 = int(round(parse_time(start) * fps)) if start is not None else 0
    f1 = min(n_total - 1, int(round(parse_time(end) * fps))) if end is not None else n_total - 1
    n = f1 + 1
    lab = {}
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            lab[int(r["Frame"])] = (int(float(r["Visibility"])), float(r["X"]), float(r["Y"]))
    done = [k for k in lab if f0 <= k <= f1]
    i = max(done) + 1 if done and max(done) < f1 else f0  # resume inside this section
    print(f"section: frames {f0}-{f1} ({f0 / fps:.1f}s-{f1 / fps:.1f}s), {len(done)} already labelled")
    win = "label_ball (click ball | n=no ball | a/d prev/next | q quit)"
    state = {"click": None}
    scale = 1.0

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / scale, y / scale)

    def save():
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Visibility", "X", "Y"])
            for k in sorted(lab):  # only frames that were actually labelled
                v, x, y = lab[k]
                w.writerow([k, v, round(x, 1), round(y, 1)])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while f0 <= i <= f1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        scale = min(1.0, 1400 / frame.shape[1])
        disp = cv2.resize(frame, None, fx=scale, fy=scale)
        for k, col in ((i - 2, (0, 120, 255)), (i - 1, (0, 200, 255))):  # previous labels as guides
            if k in lab and lab[k][0]:
                cv2.circle(disp, (int(lab[k][1] * scale), int(lab[k][2] * scale)), 6, col, 1)
        if i in lab and lab[i][0]:
            cv2.circle(disp, (int(lab[i][1] * scale), int(lab[i][2] * scale)), 8, (0, 0, 255), 2)
        txt = f"frame {i} ({i / fps:.1f}s)  section {i - f0 + 1}/{f1 - f0 + 1}  labelled {len(lab)}" + ("  [no ball]" if i in lab and not lab[i][0] else "")
        cv2.putText(disp, txt, (10, 28), 0, 0.7, (0, 0, 0), 4)
        cv2.putText(disp, txt, (10, 28), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        state["click"] = None
        while True:
            k = cv2.waitKey(20) & 0xFF
            if state["click"]:
                lab[i] = (1, *state["click"]); i += 1; break
            if k in (ord("n"), ord(" ")):
                lab[i] = (0, 0.0, 0.0); i += 1; break
            if k == ord("d"):
                i += 1; break
            if k == ord("a"):
                i = max(f0, i - 1); break
            if k == ord("f"):
                i = min(f1, i + 10); break
            if k == ord("b"):
                i = max(f0, i - 10); break
            if k == ord("s"):
                save()
            if k in (ord("q"), 27):
                save(); cv2.destroyAllWindows(); print(f"saved {out}"); return
    save()
    cv2.destroyAllWindows()
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="click-to-label ball positions")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="CSV path (default: next to the video)")
    ap.add_argument("--start", help="section start, seconds or m:ss")
    ap.add_argument("--end", help="section end, seconds or m:ss")
    a = ap.parse_args()
    main(a.video, a.out, a.start, a.end)
