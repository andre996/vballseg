# vballseg — volleyball video segmentation (CPU)

One command turns a match video into:

| Task | Output |
|---|---|
| **Rallies** (live play vs. dead time) | `rallies.csv`, `clips/rally_###.mp4`, `rallies_only.mp4` (all dead time cut out) |
| **Players & ball** (instance masks + tracking) | per-frame boxes, mask polygons, track IDs, ball position in `frames.json` |
| **Actions** (serve / receive / dig / set / attack / block / free ball) | `events.csv` with time, player ID, side, zone |
| **Court & zones** | per-frame image→court homography, each player's court position (m), side A/B and rotation zone 1–6, zone-occupancy stats in `summary.json` |
| Review | `overlay.mp4`: masks, IDs, ball trail, court lines, action labels and a live mini-map |

It's built for **mixed phone footage on a laptop CPU**: it handles variable frame rate, handheld shake and panning (camera-motion compensation), and scene cuts between clips (it resets tracking and re-detects the court).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # pulls ultralytics (CPU torch), opencv, numpy
# ffmpeg must be on PATH (clips, audio/whistle detection, H.264 overlay)
```

The first run downloads `yolo11n-seg.pt` (~6 MB).

## Use

```bash
# full pipeline
vballseg analyze match.mp4 -o out/

# a whole folder of phone clips -> out/<clip_name>/...
vballseg analyze ~/Videos/tournament/ -o out/

# faster: no masks, every 3rd frame, smaller frames
vballseg analyze match.mp4 -o out/ --no-masks --stride 3 --width 768

# only the last 3 minutes, or a section (seconds or m:ss)
vballseg analyze match.mp4 -o out/ --last 3:00
vballseg analyze match.mp4 -o out/ --start 10:00 --end 13:00

# re-cut clips with different padding, without re-analysing
vballseg clips out/summary.json --pad 2
```

### Court calibration (recommended for phone clips)

Auto court detection only works when all four boundary lines are visible and clearly painted. For anything else, click the court once:

```bash
vballseg calibrate match.mp4 --time 5 -o court.json                    # opens a window, click 4 points
vballseg calibrate match.mp4 --preset far_half -o court.json            # near baseline out of shot
vballseg calibrate match.mp4 --points "170,680 1110,680 850,250 430,250" -o court.json   # headless
vballseg analyze match.mp4 --court court.json -o out/
```

Point order for `corners`: near-left, near-right, far-right, far-left. A `_preview.jpg` is saved so you can check the fit. If the camera moves, the calibration follows it through camera-motion tracking.

**Court coordinates:** X across the court is 0–9 m, Y along it is 0–18 m, and the net is at Y = 9. Side A is Y < 9 (near the camera in an end view, left in a side view). Zones use standard rotation numbering from each team's view of the net: front row 4-3-2, back row 5-6-1.

## Ball tracking with TrackNetV3 (recommended)

The ball is tracked by **TrackNetV3**, a heatmap network that looks at 8 consecutive frames plus a background image. That lets it follow a small, motion-blurred ball much better than a single-frame detector. YOLO is still used for the players: TrackNet only tracks the ball.

1. **Get weights** (TrackNetV3 checkpoint format):
   - Beach volleyball fine-tune (MIT): `tracknet_best.pt` from huggingface.co/deadfast/beach-volley-vision-models
   - Or the original badminton weights from github.com/qaz812345/TrackNetV3 (Google Drive link in their README). This is a good *starting point* for fine-tuning, not for direct use on volleyball.
2. Put the file next to where you run the command as `TrackNet_best.pt` (or `tracknet_best.pt`, or in `weights/`). It's then used automatically, or you can pass `--ball-model path\to\file.pt`.
   The log line `ball: TrackNetV3 (...)` confirms it is active. With no weights found, the tool falls back to YOLO + the motion tracker (`--ball-model yolo` forces that).
3. **Fine-tune on your own footage** (strongly recommended for indoor phone clips):
   ```bash
   python tools/label_ball.py clip1.mp4            # click the ball frame by frame -> clip1.csv
   python tools/label_ball.py match.mp4 --start 1:25 --end 1:40   # or only a section (repeat to add more)
   python tools/train_tracknet.py --data clip1.mp4:clip1.csv clip2.mp4:clip2.csv \
          --init tracknet_best.pt --out TrackNet_volley.pt --epochs 10 --frame-step 2
   vballseg analyze match.mp4 --ball-model TrackNet_volley.pt -o out/
   python tools/eval_ball.py out/frames.json match_labels.csv   # precision / recall vs. labels
   ```
   Keep `--frame-step` equal to the `--stride` you analyse with (default 2). Start from existing weights (`--init`): training from scratch needs far more data and time. Aim for 1,500–3,000 labelled frames from the camera angles you actually film. A GPU (e.g. free Colab) makes training many times faster; the resulting `.pt` runs fine on CPU.

Speed: TrackNet costs about as much as YOLO (~9 frames/s on 2 CPU cores for TrackNet alone).

## How it works

```
frames ──► YOLO11n-seg + ByteTrack ──► players (masks, IDs)      ┐
   │                └──► ball dets ─┐                            │
   ├─► ORB camera motion ───────────┼─► motion-compensated blob  │
   │                                │   ball tracker + cleanup   ├─► rally segmentation ─► action spotting
   ├─► court lines / calibration ───┴─► homography ─► zones      │   (activity + whistle)   (touch grammar)
   └─► audio ─► whistle detector ────────────────────────────────┘
```

* **Ball**: the COCO "sports ball" class misses many volleyball frames, so it is fused with a motion-blob tracker. Blobs are found by differencing against the previous frame, warped to cancel camera motion. A track counts only once it has been confirmed over several frames, and points that wiggle in place, zig-zag or sit inside a player box (limbs) are discarded.
* **Rallies**: activity = ball moving fast + players moving (measured in body-heights/s, so zoom doesn't matter). Thresholds adapt to each clip (Otsu split). A referee whistle always marks a boundary: serve whistles extend the clip, end whistles close the rally.
* **Actions**: contacts are sharp changes in the ball's velocity with a player within reach. The contacting player is chosen by distance plus apparent size: the ball's size has to match the player's size at the same depth. Labels come from volleyball's rules: serve, then up to three contacts per side; a block doesn't count as a contact.

## Measured on the included synthetic test clip

`tools/make_synthetic.py` makes a 44 s test clip with ground truth: real people composited onto a court, 3 scripted rallies, camera shake and pan, and whistles. `tools/evaluate.py` scores a run against that ground truth:

```
rallies: predicted 3, truth 3, matched(IoU>0.5) 3, mean IoU 0.86
touches: recall 0.81, precision 0.65, action-type accuracy 0.53
court auto-detected on 100% of frames; scene cut + court re-detection verified on a spliced clip
speed: ~6.4 analysed fps on 2 CPU cores (stride 2 => ~2.3x video length)
```

## Honest limits and what to do next

1. **Ball detection is the bottleneck for actions.** The biggest gain comes from fine-tuning a small detector on volleyball frames. Label 500–2,000 frames with `ball` and `player` classes, then run `yolo segment train model=yolo11n-seg.pt data=vb.yaml imgsz=640` and pass `--model best.pt`. Class names containing "ball" and "player" are picked up automatically.
2. **Action labels are rule-based.** Side errors (mostly from players at the net hiding each other) push the type accuracy down. Correct `events.csv` by hand for a few matches, then train a temporal action spotter on ball and pose features.
3. **No court mapping** (`--court off`, or detection failed): rallies and touches still work, but actions come out as `serve` / `touch`, because side assignment needs the court.
4. **Whistles** help a lot when the referee is audible. In noisy gyms, use `--no-whistle` if you get false splits.
5. Tune rally sensitivity with `--rally-high` / `--rally-low` / `--min-rally`. Each rally in `summary.json` has a `confidence` you can filter on.

## Output schema (short)

`summary.json` → `rallies[] {id, start, end, core_start, core_end, confidence, cues}`, `events[] {rally, t, type, confidence, player_id, side, zone, court_xy}`, `stats {zone_occupancy_pct, actions_by_zone, action_counts}`, `analysis {court_coverage, scene_cuts, whistles}`.

`frames.json` → `frames[] {frame, t, shot, ball {xy, src}, players[] {id, box, mask, foot, court, side, zone}, H_court2img}`. Pixel coordinates are at analysis resolution; divide by `scale` to get original-video pixels.
