"""Score ball tracking from a vballseg run against TrackNet-format labels.

    python tools/eval_ball.py out/frames.json clip.csv [--from 30] [--tol 15]

A prediction counts as correct if it is within --tol pixels (original resolution) of the label.
"""
import argparse
import csv
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_json")
    ap.add_argument("labels_csv")
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--from", dest="t0", type=float, default=0.0, help="only score frames after this time (s)")
    a = ap.parse_args()
    F = json.load(open(a.frames_json))
    s = F["scale"]
    lab = {int(r["Frame"]): r for r in csv.DictReader(open(a.labels_csv))}
    tp = fp = fn = tn = n = nvis = 0
    errs = []
    for f in F["frames"]:
        if f["t"] < a.t0 or f["frame"] not in lab:
            continue
        r = lab[f["frame"]]
        vis = int(float(r["Visibility"])) > 0
        n += 1
        nvis += vis
        b = f.get("ball")
        if b and b.get("src") == "interp":
            b = None  # score raw detections only
        if vis and b:
            d = float(np.hypot(b["xy"][0] / s - float(r["X"]), b["xy"][1] / s - float(r["Y"])))
            errs.append(d)
            if d <= a.tol:
                tp += 1
            else:
                fp += 1; fn += 1
        elif vis:
            fn += 1
        elif b:
            fp += 1
        else:
            tn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"frames scored: {n}  ball visible in {nvis}")
    print(f"precision {prec:.2f}  recall {rec:.2f}  F1 {2 * prec * rec / max(1e-9, prec + rec):.2f}  "
          f"median error {np.median(errs) if errs else float('nan'):.1f}px")


if __name__ == "__main__":
    main()
