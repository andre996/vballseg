"""Compare a vballseg summary.json against ground_truth.json (from make_synthetic.py or hand labels).

ground truth format: {"rallies": [{"start": s, "end": s}], "events": [{"t": s, "type": str, "side": "A"|"B"}]}
"""
import json
import sys


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def main(summary, truth, tol=0.35):
    s, g = json.load(open(summary)), json.load(open(truth))
    pr = [(r["core_start"], r["core_end"]) for r in s["rallies"]]
    gt = [(r["start"], r["end"]) for r in g["rallies"]]
    matched = [max((iou(p, q) for p in pr), default=0) for q in gt]
    tp = sum(m > 0.5 for m in matched)
    print(f"rallies: predicted {len(pr)}, truth {len(gt)}, matched(IoU>0.5) {tp}, "
          f"mean IoU {sum(matched) / max(1, len(gt)):.2f}")
    for q, m in zip(gt, matched):
        print(f"   truth {q[0]:6.2f}-{q[1]:6.2f}  best IoU {m:.2f}")
    ev, used = s["events"], set()
    hit = typ_ok = 0
    for e in g["events"]:
        best = None
        for i, p in enumerate(ev):
            if i in used:
                continue
            d = abs(p["t"] - e["t"])
            if d <= tol and (best is None or d < best[0]):
                best = (d, i)
        if best:
            used.add(best[1]); hit += 1
            typ_ok += ev[best[1]]["type"] == e["type"]
    n = len(g["events"])
    print(f"touches: truth {n}, predicted {len(ev)}, recall {hit / max(1, n):.2f}, "
          f"precision {hit / max(1, len(ev)):.2f}, action-type accuracy on matched {typ_ok / max(1, hit):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
