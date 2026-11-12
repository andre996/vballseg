"""Outputs: JSON/CSV, per-rally clips, highlight reel, zone statistics."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List


def write_json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def write_csvs(out_dir: str, rallies: List[Dict], events: List[Dict]) -> None:
    with open(os.path.join(out_dir, "rallies.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rally", "start_s", "end_s", "duration_s", "confidence", "cues", "n_touches"])
        n = Counter(e["rally"] for e in events)
        for r in rallies:
            w.writerow([r["id"], r["start"], r["end"], round(r["end"] - r["start"], 2), r["confidence"],
                        "|".join(r["cues"]), n.get(r["id"], 0)])
    with open(os.path.join(out_dir, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rally", "time_s", "action", "confidence", "player_id", "side", "zone", "court_x", "court_y"])
        for e in events:
            c = e.get("court_xy") or [None, None]
            w.writerow([e["rally"], e["t"], e["type"], e["confidence"], e["player_id"], e.get("side"),
                        e.get("zone"), c[0], c[1]])


def zone_stats(frames: List[Dict], rallies: List[Dict], events: List[Dict]) -> Dict:
    """Player-time per zone during live play, and action counts by zone/side."""
    occ: Dict[str, Counter] = defaultdict(Counter)
    spans = [(r["core_start"], r["core_end"]) for r in rallies]
    for f in frames:
        if not any(a <= f["t"] <= b for a, b in spans):
            continue
        for p in f["players"]:
            if p.get("side") in ("A", "B") and p.get("zone") is not None:
                occ[p["side"]][str(p["zone"])] += 1
    acts: Dict[str, Counter] = defaultdict(Counter)
    for e in events:
        acts[f"{e.get('side')}:{e['type']}"][str(e.get("zone"))] += 1
    tot = {s: sum(c.values()) for s, c in occ.items()}
    return {"zone_occupancy_pct": {s: {z: round(100 * n / tot[s], 1) for z, n in sorted(c.items())}
                                   for s, c in occ.items()},
            "actions_by_zone": {k: dict(v) for k, v in acts.items()},
            "action_counts": dict(Counter(e["type"] for e in events))}


def cut_clips(video: str, rallies: List[Dict], out_dir: str, reel: bool = True) -> List[str]:
    clip_dir = os.path.join(out_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    paths = []
    for r in rallies:
        p = os.path.join(clip_dir, f"rally_{r['id']:03d}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{r['start']:.2f}", "-i", video,
               "-t", f"{r['end'] - r['start']:.2f}", "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", p]
        subprocess.run(cmd, check=True)
        paths.append(p)
    if reel and paths:
        lst = os.path.join(clip_dir, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{os.path.abspath(p)}'\n" for p in paths)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", os.path.join(out_dir, "rallies_only.mp4")], check=True)
        os.remove(lst)
    return paths
