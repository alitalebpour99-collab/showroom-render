"""Independent beat/cut check of a FINISHED Reel (does not trust the render plan).

usage: python sync_check.py <reel.mp4> <render-log.json> <out_dir>

- beats: comb fit of the kick band of the reel's own audio (music + sound design), tempo searched +-5 % of the plan;
- cuts: grey-histogram + coarse-layout change peaks (beat zoom pulses do not count, whips and flashes do);
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beatgrid  # noqa: E402

mp4, logf, out = sys.argv[1], sys.argv[2], sys.argv[3]
log = json.load(open(logf, encoding="utf-8"))
T = log["template"]
period = log["music"]["period"]
FPS = 30

y, sr = librosa.load(mp4, sr=beatgrid.SRA, mono=True)
dur = len(y) / sr
on, hop_s = beatgrid.kick_onsets(y.astype(np.float64), sr)
bpm0 = 60 / period
bpm_m, b1, phase, contrast = beatgrid.comb(on, hop_s, 0.0, dur - 0.05, bpm0 * 0.95, bpm0 * 1.05, step=0.01, ph_step=hop_s)
phase = phase % b1
if phase > b1 / 2:  # a beat just before 0 is the beat at 0
    phase -= b1
beats = np.arange(phase, dur + 1e-6, b1)
eighths = np.sort(np.concatenate([beats, beats[:-1] + b1 / 2]))

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vf", "scale=96:170,format=gray", "-f", "rawvideo", "-"],
                     capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, 170, 96).astype(np.float32)
hist = np.stack([np.histogram(f, bins=32, range=(0, 256))[0] for f in fr]).astype(np.float32) / (170 * 96)
small = np.stack([cv2.resize(f, (12, 21), interpolation=cv2.INTER_AREA) for f in fr])
dh = 0.5 * np.abs(np.diff(hist, axis=0)).sum(1)
dl = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
score = np.maximum(dh / 0.18, dl / 16.0)
cuts = []
for i in range(len(score)):
    if score[i] >= 1.0 and score[i] == score[max(0, i - 2):i + 3].max():
        t = (i + 1) / FPS  # the new picture starts at frame i+1
        if not cuts or t - cuts[-1] > 0.2:
            cuts.append(t)
pc = [c for c in log["cuts"] if c["t0"] > 0.01]
plan = [c["t0"] for c in pc]
loop_from = dur - 6 / FPS - 0.05  # the closing dissolve into frame 0 is not a cut
rows = []
for t in cuts:
    e = float(np.min(np.abs(eighths - t)))
    c = min(pc, key=lambda x: abs(x["t0"] - t)) if pc else None
    kind = "loop" if t >= loop_from else ("whip" if c and c.get("tr") == "whip" and abs(c["t0"] - t) < 0.15 else "cut")
    rows.append(dict(t=round(t, 3), kind=kind, beat_err_ms=round(1000 * e, 1), plan_cut=round(c["t0"], 3) if c else None,
                     plan_err_ms=round(1000 * abs(c["t0"] - t), 1) if c else None))
hard = [r["beat_err_ms"] for r in rows if r["kind"] == "cut"]
missed = [round(p["t0"], 3) for p in pc if not any(abs(p["t0"] - t) < (0.15 if p.get("tr") == "whip" else 0.08) for t in cuts)]
drop = log["sync_plan"]["drop_t"]
res = dict(template=T, bpm_measured=round(60 / b1, 2), grid_contrast=round(contrast, 2), bpm_plan=log["music"]["bpm"],
           beat_phase_ms=round(1000 * phase, 1),
           cuts_detected=len(cuts), cuts_planned=len(plan), planned_cuts_not_seen=missed,
           max_beat_err_ms=max(hard or [0]),
           mean_beat_err_ms=round(float(np.mean(hard or [0])), 1),
           whip_onsets_ms=[r["beat_err_ms"] for r in rows if r["kind"] == "whip"],
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
