"""Independent beat/cut check of a FINISHED Reel (does not trust the render plan).

usage: python sync_check.py <reel.mp4> <render-log.json> <out_dir>

- beats: librosa on the reel's own audio (music + sound design), fitted to a straight line with the bpm from the log;
- cuts: frame-difference peaks of the decoded video (small grey frames), so hard cuts, whips and flashes all count;
- every cut gets its distance to the nearest beat / eighth; the drop must carry a cut;
- writes sync-check-<t>.json and sync-check-<t>.png (waveform, beats, bar lines in gold, cuts in red).
"""
import json
import os
import subprocess
import sys

import cv2
import librosa
import numpy as np

mp4, logf, out = sys.argv[1], sys.argv[2], sys.argv[3]
log = json.load(open(logf, encoding="utf-8"))
T = log["template"]
period = log["music"]["period"]
FPS = 30

y, sr = librosa.load(mp4, sr=22050, mono=True)
dur = len(y) / sr
env = librosa.onset.onset_strength(y=y, sr=sr)
_, bfr = librosa.beat.beat_track(onset_envelope=env, sr=sr, start_bpm=60 / period, tightness=300)
bt = librosa.frames_to_time(bfr, sr=sr)
idx = np.round((bt - bt[0]) / period)
b1, b0 = np.polyfit(idx, bt, 1) if len(bt) > 3 else (period, 0.0)
phase = b0 % b1
beats = np.arange(phase, dur + 1e-6, b1)
# downbeats from the plan (the reel starts on a downbeat by construction); report the measured phase offset
eighths = np.sort(np.concatenate([beats, beats[:-1] + b1 / 2]))

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vf", "scale=96:170,format=gray", "-f", "rawvideo", "-"],
                     capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, 170, 96).astype(np.float32)
d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
med = float(np.median(d)) + 1e-3
cuts = []
for i in range(1, len(d) - 1):
    if d[i] >= d[i - 1] and d[i] >= d[i + 1] and d[i] > max(6.0, 3.2 * med):
        t = (i + 1) / FPS  # the new picture starts at frame i+1
        if not cuts or t - cuts[-1] > 0.15:
            cuts.append(t)
plan = [c["t0"] for c in log["cuts"] if c["t0"] > 0.01]
rows = []
for t in cuts:
    e = float(np.min(np.abs(eighths - t)))
    p = min(plan, key=lambda x: abs(x - t)) if plan else None
    rows.append(dict(t=round(t, 3), beat_err_ms=round(1000 * e, 1), plan_cut=round(p, 3) if p is not None else None,
                     plan_err_ms=round(1000 * abs(p - t), 1) if p is not None else None))
missed = [round(p, 3) for p in plan if not any(abs(p - t) < 0.08 for t in cuts)]
drop = log["sync_plan"]["drop_t"]
res = dict(template=T, bpm_measured=round(60 / b1, 2), bpm_plan=log["music"]["bpm"],
           beat_phase_ms=round(1000 * (phase if phase < b1 / 2 else phase - b1), 1),
           cuts_detected=len(cuts), cuts_planned=len(plan), planned_cuts_not_seen=missed,
           max_beat_err_ms=max([r["beat_err_ms"] for r in rows] or [0]),
           mean_beat_err_ms=round(float(np.mean([r["beat_err_ms"] for r in rows] or [0])), 1),
           drop_t=drop, drop_has_cut=any(abs(t - drop) < 0.05 for t in cuts), cuts=rows,
           loop=dict(first_last_frame_diff=round(float(np.abs(fr[0] - fr[-1]).mean()), 1)))
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, f"sync-check-{T}.json"), "w"), indent=1)

# picture: 1600 x 360
Wd, Hd = 1600, 360
img = np.full((Hd, Wd, 3), 18, np.uint8)
X = lambda t: int(20 + (Wd - 40) * t / dur)
hop = len(y) // (Wd - 40)
for k in range(Wd - 40):
    a = float(np.abs(y[k * hop:(k + 1) * hop]).max()) if hop else 0
    cv2.line(img, (20 + k, Hd // 2 - int(a * 120)), (20 + k, Hd // 2 + int(a * 120)), (110, 110, 110), 1)
for i, b in enumerate(beats):
    gold = i % 4 == 0
    cv2.line(img, (X(b), 30), (X(b), Hd - 30), (84, 172, 214) if gold else (70, 70, 70), 2 if gold else 1)
for t in cuts:
    cv2.line(img, (X(t), 10), (X(t), 60), (60, 60, 235), 3)
cv2.line(img, (X(drop), Hd - 30), (X(drop), Hd - 4), (255, 255, 255), 3)
cv2.putText(img, f"{T}  bpm {res['bpm_measured']}  cuts {len(cuts)}/{len(plan)}  max err {res['max_beat_err_ms']} ms  mean {res['mean_beat_err_ms']} ms  drop cut {res['drop_has_cut']}",
            (24, Hd - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(out, f"sync-check-{T}.png"), img)
print(json.dumps({k: v for k, v in res.items() if k != "cuts"}, indent=1))
