"""Taleb Pour Car City - template 2 «جزئیات روی بیت» (details on the beat), 10 s demo.

Pipeline (all free, CPU, GitHub Actions):
  pick Setavaz track + 10 s window by beat strength (librosa)
  -> cut list snapped to beats (opener 2 beats, details 1 beat, hero 2 beats, end card)
  -> per shot: ffmpeg decode + house LUT + upscale to 1080x1920
  -> speed ramps (time remap), dark-sky graduated filter, gold light sweep + glint,
     plate detection (open-image-models YOLOv9) + IoU tracking + branded plate graphic,
     beat punch-zoom, model label, watermark, follow badge, end card
  -> code-made sound design (door thud, whoosh, shimmer, badge cue) + music, loudnorm
"""
import json
import math
import os
import subprocess
import sys

import cv2
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

RAQM = features.check("raqm")

import arabic_reshaper
from bidi.algorithm import get_display

W, H, FPS = 1080, 1920, 30
TOTAL = 10.0
END_LEN = 1.5
ROOT = sys.argv[1] if len(sys.argv) > 1 else "work"
IN = os.path.join(ROOT, "in")
OUT = os.path.join(ROOT, "out")
LUT = os.path.join(ROOT, "golden-gallery.cube")
FONTS = os.path.join(ROOT, "fonts")
os.makedirs(OUT, exist_ok=True)

PHONE = "۰۹۱۲ ۰۰۰ ۰۰۰۰"          # PLACEHOLDER until Ali gives the real one
HANDLE = "@talebpour.carcity"      # PLACEHOLDER
GOLD = (214, 172, 84)
IVORY = (246, 236, 211)
CREAM = (243, 233, 210)

LOG = {}


def fa(s):
    # Pillow built with libraqm shapes + reorders Persian itself; only fall back to manual reshaping without it
    return s if RAQM else get_display(arabic_reshaper.reshape(s))


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, f"Vazirmatn-{name}.ttf"), size)


# ----------------------------------------------------------------- shots
def S(src, a, b, group=None, label=None, sweep=False, sky=False, ramp=False):
    return dict(src=src, a=a, b=b, group=group, label=label, sweep=sweep, sky=sky, ramp=ramp)


OPENER = S("B", 0.20, 2.85, "m1", "MERCEDES-BENZ", sweep=True, sky=True, ramp=True)
MIDDLE = [
    S("D", 2.72, 3.35, "g", "GENESIS G80"),   # bonnet badge
    S("D", 3.45, 4.05, "g"),                  # headlight
    S("D", 6.32, 6.95, "g"),                  # GENESIS rear script
    S("D", 7.05, 7.68, "g"),                  # steering wheel
    S("D", 8.52, 9.15, "g"),                  # beige seat
    S("D", 9.25, 9.88, "g"),                  # console
    S("A", 6.62, 7.22, "k", "KIA"),           # kia wheel
    S("B", 4.38, 6.15, "c3", "C 300", sky=True),
    S("A", 9.45, 10.28, "amg", "MERCEDES-AMG", sky=True),
    S("A", 2.88, 3.68, "j", "JAGUAR XF", sky=True),
    S("A", 7.32, 7.98, "h"),
    S("A", 8.08, 8.65, "t"),
]
HERO = S("C", 1.08, 2.10, "m2", "MERCEDES-BENZ  C-CLASS", sweep=True, sky=True, ramp=True)
ENDC = S("B", 2.95, 4.28, "end", sky=True)


# ----------------------------------------------------------------- music
def pick_music():
    cands = sorted(f for f in os.listdir(os.path.join(IN, "music")) if f.endswith((".mp3", ".flac", ".wav")))
    best = None
    for f in cands:
        y, sr = librosa.load(os.path.join(IN, "music", f), sr=22050, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        bt = librosa.frames_to_time(beats, sr=sr)
        pulse = float(env[beats].mean() / (env.mean() + 1e-6)) if len(beats) else 0
        rms = librosa.feature.rms(y=y)[0]
        rt = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        # best 10 s window starting on a beat, after the intro
        win = None
        for i, t0 in enumerate(bt):
            if t0 < 8 or t0 + TOTAL + 0.5 > len(y) / sr:
                continue
            m = (rt >= t0) & (rt < t0 + TOTAL)
            e = float(rms[m].mean())
            if win is None or e > win[1]:
                win = (t0, e, i)
        if win is None:
            continue
        eff = tempo if 95 <= tempo <= 150 else (tempo * 2 if tempo < 95 else tempo / 2)
        score = pulse * win[1] * (1.0 if 100 <= eff <= 135 else 0.7)
        info = dict(file=f, tempo=round(tempo, 1), pulse=round(pulse, 2), start=round(float(win[0]), 3),
                    energy=round(win[1], 4), score=round(score, 4))
        print("music cand", info)
        if best is None or score > best[0]["score"]:
            rel = [float(t - win[0]) for t in bt if win[0] - 1e-3 <= t < win[0] + TOTAL]
            best = (info, rel)
    LOG["music"] = best[0]
    return best


def cut_plan(beats):
    period = float(np.median(np.diff(beats)))
    step = 1 if period >= 0.42 else 2  # keep cuts >= ~0.42 s
    grid = beats[::step]
    LOG["beat_period"] = round(period, 3)
    target_end = TOTAL - END_LEN
    # montage end = grid point closest to TOTAL-END_LEN
    end_i = int(np.argmin([abs(t - target_end) for t in grid]))
    n = end_i  # number of grid intervals in montage
    plan = []
    idx = 0

    def add(shot, k):
        nonlocal idx
        plan.append(dict(shot, t0=grid[idx], t1=grid[idx + k]))
        idx += k

    add(OPENER, 2)
    mids = max(0, n - 4)
    use = MIDDLE[:mids]
    ks = [1] * len(use)
    extra = mids - len(use)
    pref = [i for i in (7, 8, 9, 0, 4, 3) if i < len(use)]
    j = 0
    while extra > 0 and pref:
        ks[pref[j % len(pref)]] += 1
        extra -= 1
        j += 1
    for s, k in zip(use, ks):
        add(s, k)
    add(HERO, 2)
    plan.append(dict(ENDC, t0=grid[idx], t1=TOTAL))
    plan[0]["t0"] = 0.0
    return plan, grid


# ----------------------------------------------------------------- video io
SRC = {"A": "A.mp4", "B": "B.mp4", "C": "C.mp4", "D": "D.mp4"}


def decode(shot):
    src = os.path.join(IN, SRC[shot["src"]])
    dur = shot["b"] - shot["a"]
    vf = f"lut3d={LUT},scale={W}:{H}:flags=lanczos,format=bgr24"
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{shot['a']:.3f}", "-t", f"{dur:.3f}", "-i", src,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return fr


def remap(frames, n_out, ramp):
    m = len(frames)
    out = []
    for i in range(n_out):
        u = i / max(1, n_out - 1)
        if ramp:
            # fast in, slow out: speed 1.7x -> 0.3x of average
            p = 1.7 * u - 0.7 * u * u
        else:
            p = u
        p = min(p, 1.0)
        if not ramp and m >= n_out:
            src_i = (m - n_out) // 2 + i      # real-time, centred trim
        else:
            src_i = int(round(p * (m - 1)))  # ramp, or gentle stretch of a short source
        out.append(frames[min(m - 1, max(0, src_i))])
    return out


# ----------------------------------------------------------------- effects
_ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]


def sky_filter(f, strength=0.42):
    reach = 0.40
    g = np.clip(1 - _ys / reach, 0, 1) ** 1.6
    img = f.astype(np.float32)
    lum = img.mean(-1, keepdims=True) / 255.0
    # darken mostly the bright, flat sky; tint towards deep blue-grey
    k = (g[..., None] if g.ndim == 2 else g) * strength * np.clip((lum - 0.35) / 0.5, 0, 1)
    k = k.reshape(H, W, 1) if k.ndim == 3 else k
    tint = np.array([120, 100, 88], np.float32)  # BGR deep slate
    img = img * (1 - k) + tint * k * 0.55
    return np.clip(img, 0, 255).astype(np.uint8)


_xx, _yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def light_sweep(f, u, glint_pt=None):
    """u in [0,1] progress of the sweep across the frame."""
    if u < 0 or u > 1:
        return f
    ang = math.radians(-32)
    d = _xx * math.cos(ang) + _yy * math.sin(ang)
    dmin, dmax = d.min(), d.max()
    pos = dmin + (dmax - dmin) * (-0.15 + 1.3 * u)
    wband = 0.06 * (dmax - dmin)
    band = np.exp(-((d - pos) / wband) ** 2)
    img = f.astype(np.float32) / 255.0
    lum = img.mean(-1)
    mask = np.clip((lum - 0.40) / 0.45, 0, 1) ** 1.4
    a = (band * (0.18 + 0.62 * mask))[..., None] * 0.85
    col = np.array([205, 238, 255], np.float32) / 255.0  # BGR warm white-gold
    img = 1 - (1 - img) * (1 - a * col)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if glint_pt is not None:
        env = math.exp(-((u - 0.55) / 0.12) ** 2)
        if env > 0.03:
            out = glint(out, glint_pt, env)
    return out


def glint(f, pt, env):
    x0, y0 = pt
    r = 150
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    rr = np.sqrt(xs ** 2 + ys ** 2) + 1e-3
    core = np.exp(-(rr / 9) ** 2)
    ray = np.exp(-(np.abs(ys) / 2.2) ** 2) * np.exp(-np.abs(xs) / 60) + np.exp(-(np.abs(xs) / 2.2) ** 2) * np.exp(-np.abs(ys) / 60)
    diag = (np.exp(-(np.abs(xs - ys) / 2.5) ** 2) + np.exp(-(np.abs(xs + ys) / 2.5) ** 2)) * np.exp(-rr / 25) * 0.5
    s = np.clip(core + 0.9 * ray + diag, 0, 1) * env
    x1, y1 = max(0, x0 - r), max(0, y0 - r)
    x2, y2 = min(W, x0 + r + 1), min(H, y0 + r + 1)
    if x2 <= x1 or y2 <= y1:
        return f
    sub = s[y1 - (y0 - r):y2 - (y0 - r), x1 - (x0 - r):x2 - (x0 - r)][..., None]
    reg = f[y1:y2, x1:x2].astype(np.float32) / 255.0
    col = np.array([210, 240, 255], np.float32) / 255.0
    reg = 1 - (1 - reg) * (1 - sub * col)
    f = f.copy()
    f[y1:y2, x1:x2] = (np.clip(reg, 0, 1) * 255).astype(np.uint8)
    return f


def glint_point(frame, plates):
    """brightest small highlight in the central area (chrome / star), away from plates."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 6).astype(np.float32)
    m = np.zeros_like(g)
    m[int(H * 0.25):int(H * 0.75), int(W * 0.2):int(W * 0.8)] = 1
    for (x1, y1, x2, y2) in plates:
        m[max(0, y1 - 40):y2 + 40, max(0, x1 - 40):x2 + 40] = 0
    g = g * m
    y, x = np.unravel_index(np.argmax(g), g.shape)
    return int(x), int(y)


def punch(f, k, n=5, amt=0.045):
    if k >= n:
        return f
    s = 1 + amt * (1 - k / n) ** 2
    nw, nh = int(W / s), int(H / s)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    return cv2.resize(f[y0:y0 + nh, x0:x0 + nw], (W, H), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------- plates
_det = None
REJECT = []


def detector():
    global _det
    if _det is None:
        from open_image_models import create_detector
        _det = create_detector("yolo-v9-s-608-license-plate-end2end")
    return _det


def detect(frames):
    res = []
    for f in frames:
        boxes = []
        for d in detector().predict(f):
            b = d.bounding_box
            bw, bh = b.x2 - b.x1, b.y2 - b.y1
            ar = bw / max(1, bh)
            if d.confidence < 0.45 or not (1.6 <= ar <= 7.5) or bw < 24:
                REJECT.append(dict(conf=round(float(d.confidence), 2), ar=round(ar, 2), w=int(bw)))
                continue
            boxes.append((b.x1, b.y1, b.x2, b.y2, float(d.confidence)))
        res.append(boxes)
    return res


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def track(dets, n):
    tracks = []  # each: dict(frames={i: box})
    for i, boxes in enumerate(dets):
        for bx in boxes:
            best, bi = None, 0.0
            for t in tracks:
                last_i = max(t["f"])
                if i - last_i > 10:
                    continue
                v = iou(t["f"][last_i], bx)
                cx = abs((t["f"][last_i][0] + t["f"][last_i][2]) - (bx[0] + bx[2])) / 2
                if v > bi or (v == 0 and cx < 80 and best is None):
                    best, bi = t, v
            if best is not None and i not in best["f"]:
                best["f"][i] = bx
            else:
                tracks.append({"f": {i: bx}})
    out = [[] for _ in range(n)]
    kept = 0
    for t in tracks:
        ks = sorted(t["f"])
        conf = max(t["f"][k][4] for k in ks)
        if len(ks) < 2 and conf < 0.6:
            continue
        kept += 1
        arr = np.array([t["f"][k][:4] for k in ks], np.float32)
        lo, hi = ks[0], ks[-1]
        # extend 6 frames before/after (plate entering/leaving the detector's reach)
        lo2, hi2 = max(0, lo - 6), min(n - 1, hi + 6)
        full = np.stack([np.interp(np.arange(lo2, hi2 + 1), ks, arr[:, j]) for j in range(4)], 1)
        # smooth
        k = 5
        pad = np.pad(full, ((k // 2, k // 2), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(4)], 1)
        for j, fi in enumerate(range(lo2, hi2 + 1)):
            out[fi].append(tuple(int(v) for v in sm[j]))
    return out, kept


def make_plate_art():
    pw, ph = 1040, 228
    im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=26, fill=CREAM + (255,), outline=GOLD + (255,), width=14)
    d.rounded_rectangle([22, 22, pw - 23, ph - 23], radius=16, outline=(170, 128, 52, 255), width=3)
    f1 = font("Bold", 104)
    t = fa("طالب‌پور")
    bb = d.textbbox((0, 0), t, font=f1)
    tw = bb[2] - bb[0]
    d.text((pw * 0.60 - tw / 2, ph / 2 - (bb[3] + bb[1]) / 2 - 6), t, font=f1, fill=(150, 108, 38, 255))
    f2 = font("Medium", 40)
    for k, line in enumerate(["CAR", "CITY"]):
        bb = d.textbbox((0, 0), line, font=f2)
        d.text((150 - (bb[2] - bb[0]) / 2, ph / 2 - 52 + k * 52), line, font=f2, fill=(40, 34, 26, 255))
    d.line([(262, 50), (262, ph - 50)], fill=GOLD + (255,), width=4)
    return np.array(im)


PLATE_ART = None


def put_plates(f, boxes):
    global PLATE_ART
    if PLATE_ART is None:
        PLATE_ART = make_plate_art()
    out = f
    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 4:
            continue
        px, py = int(bw * 0.10) + 3, int(bh * 0.22) + 3
        X1, Y1, X2, Y2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        tw, th = X2 - X1, Y2 - Y1
        if tw < 4 or th < 4:
            continue
        art = cv2.resize(PLATE_ART, (tw, th), interpolation=cv2.INTER_AREA)
        a = art[..., 3:4].astype(np.float32) / 255.0
        rgb = art[..., :3][..., ::-1].astype(np.float32)
        # match scene brightness a little so it sits in the shot
        reg = out[Y1:Y2, X1:X2].astype(np.float32)
        sc = np.clip(reg.mean() / 170.0, 0.65, 1.05)
        if out is f:
            out = f.copy()
        out[Y1:Y2, X1:X2] = np.clip(reg * (1 - a) + rgb * sc * a, 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------- overlays (RGBA)
def spaced(d, xy, text, fnt, fill, spacing):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += d.textlength(ch, font=fnt) + spacing
    return x


def spaced_len(d, text, fnt, spacing):
    return sum(d.textlength(ch, font=fnt) + spacing for ch in text) - spacing


def label_layer(text):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font("Light", 40)
    x0, y0 = 84, int(H * 0.165)
    d.line([(x0, y0 - 22), (x0 + 70, y0 - 22)], fill=GOLD + (255,), width=3)
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sh)
    spaced(ds, (x0 + 2, y0 + 2), text, f, (0, 0, 0, 160), 7)
    sh = sh.filter(ImageFilter.GaussianBlur(4))
    im = Image.alpha_composite(sh, im)
    d = ImageDraw.Draw(im)
    spaced(d, (x0, y0), text, f, IVORY + (255,), 7)
    return np.array(im)


def watermark_layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font("Medium", 24)
    t = "TALEB POUR  CAR CITY"
    tw = spaced_len(d, t, f, 6)
    spaced(d, ((W - tw) / 2, 128), t, f, GOLD + (205,), 6)
    return np.array(im)


def badge_layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x0, y0, bw, bh = 84, int(H * 0.72), 470, 74
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=37, fill=(14, 12, 10, 170), outline=GOLD + (230,), width=2)
    # heart
    cx, cy = x0 + 42, y0 + bh // 2
    d.ellipse([cx - 16, cy - 13, cx, cy + 3], fill=GOLD + (255,))
    d.ellipse([cx - 2, cy - 13, cx + 14, cy + 3], fill=GOLD + (255,))
    d.polygon([(cx - 16, cy - 3), (cx + 14, cy - 3), (cx - 1, cy + 16)], fill=GOLD + (255,))
    d.text((x0 + 74, y0 + 18), HANDLE, font=font("Medium", 28), fill=IVORY + (255,))
    t = fa("+ دنبال کنید")
    fb = font("Bold", 28)
    tw = d.textlength(t, font=fb)
    d.text((x0 + bw - 24 - tw, y0 + 14), t, font=fb, fill=GOLD + (255,))
    return np.array(im)


def endcard_layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # bottom gradient for legibility
    grad = np.zeros((H, W, 4), np.uint8)
    a = (np.clip((_ys[:, 0] - 0.35) / 0.4, 0, 1) ** 1.2 * 215).astype(np.uint8)
    grad[..., 3] = a[:, None]
    im = Image.alpha_composite(Image.fromarray(grad, "RGBA"), im)
    d = ImageDraw.Draw(im)
    cy = int(H * 0.62)
    d.line([(W / 2 - 60, cy - 70), (W / 2 + 60, cy - 70)], fill=GOLD + (255,), width=3)
    t = fa("شهرک اتومبیل طالب‌پور")
    f = font("Bold", 78)
    tw = d.textlength(t, font=f)
    d.text(((W - tw) / 2, cy - 40), t, font=f, fill=IVORY + (255,))
    f2 = font("Light", 30)
    e = "TALEB POUR CAR CITY"
    ew = spaced_len(d, e, f2, 9)
    spaced(d, ((W - ew) / 2, cy + 82), e, f2, GOLD + (255,), 9)
    t3 = fa("تماس / دایرکت")
    f3 = font("Medium", 50)
    tw = d.textlength(t3, font=f3)
    d.text(((W - tw) / 2, cy + 160), t3, font=f3, fill=GOLD + (255,))
    f4 = font("Medium", 44)
    tw = d.textlength(PHONE, font=f4)
    d.text(((W - tw) / 2, cy + 240), fa(PHONE), font=f4, fill=IVORY + (255,))
    f5 = font("Light", 32)
    tw = d.textlength(HANDLE, font=f5)
    d.text(((W - tw) / 2, cy + 310), HANDLE, font=f5, fill=IVORY + (220,))
    return np.array(im)


def comp(f, layer, alpha=1.0):
    if alpha <= 0:
        return f
    a = layer[..., 3:4].astype(np.float32) / 255.0 * alpha
    rgb = layer[..., :3][..., ::-1].astype(np.float32)
    return np.clip(f.astype(np.float32) * (1 - a) + rgb * a, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- audio (code-made SFX)
SR = 48000


def sfx_thud():
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (52 + 30 * np.exp(-t * 30)) * t) * np.exp(-t * 11)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    # low-passed click
    a = np.exp(-2 * np.pi * 900 / SR)
    lp = np.zeros(n)
    for i in range(1, n):
        lp[i] = (1 - a) * noise[i] + a * lp[i - 1]
    click = lp * np.exp(-t * 60) * 3.0
    latch = np.sin(2 * np.pi * 1800 * t) * np.exp(-((t - 0.012) / 0.004) ** 2) * 0.15
    s = body * 0.9 + click + latch
    return s / np.abs(s).max() * 0.9


def sfx_whoosh(dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n)
    env = np.sin(np.pi * (t / dur) ** 1.6) ** 2
    out = np.zeros(n)
    y1 = 0
    for i in range(n):
        fc = 300 + 3500 * (t[i] / dur) ** 2
        a = np.exp(-2 * np.pi * fc / SR)
        y1 = (1 - a) * x[i] + a * y1
        out[i] = y1
    out = out * env
    return out / np.abs(out).max() * 0.55


def sfx_shimmer(dur=0.7):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for k, fr in enumerate([2637, 3136, 3951, 5274]):
        s += np.sin(2 * np.pi * fr * t + k) * np.exp(-((t - 0.12 + 0.04 * k) / 0.18) ** 2)
    s *= 0.12
    return s


def sfx_cue():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    s = (np.sin(2 * np.pi * 1318.5 * t) * (t < 0.12) + np.sin(2 * np.pi * 1760 * t) * (t >= 0.09)) * np.exp(-t * 9)
    return s * 0.18


def place(buf, s, t):
    i = int(t * SR)
    j = min(len(buf), i + len(s))
    if j > i:
        buf[i:j] += s[:j - i]


# ----------------------------------------------------------------- main
def main():
    (minfo, beats) = pick_music()
    beats = np.array(beats)
    plan, grid = cut_plan(beats)
    LOG["cuts"] = [dict(src=p["src"], a=p["a"], b=p["b"], t0=round(p["t0"], 3), t1=round(p["t1"], 3),
                        label=p.get("label")) for p in plan]
    print(json.dumps(LOG, indent=1, ensure_ascii=False))

    wm = watermark_layer()
    badge = badge_layer()
    endl = endcard_layer()
    labels = {}
    group_start = {}
    for p in plan:
        g = p["group"]
        if g not in group_start:
            group_start[g] = p["t0"]
            if p.get("label"):
                labels[g] = label_layer(p["label"])
    group_dur = {}
    for p in plan:
        group_dur[p["group"]] = group_dur.get(p["group"], 0) + (p["t1"] - p["t0"])

    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "17", "-pix_fmt", "yuv420p", os.path.join(OUT, "video.mp4")],
                           stdin=subprocess.PIPE)
    total_frames = int(round(TOTAL * FPS))
    written = 0
    plate_log = []
    sweep_times = []
    stills = {}
    for si, p in enumerate(plan):
        f0 = int(round(p["t0"] * FPS))
        f1 = total_frames if si == len(plan) - 1 else int(round(p["t1"] * FPS))
        n = f1 - f0
        src = decode(p)
        frames = remap(list(src), n, p["ramp"])
        dets = detect(frames)
        boxes, kept = track(dets, n)
        plate_log.append(dict(shot=si, src=p["src"], frames=n, detected_frames=sum(1 for d in dets if d),
                              sample=[list(map(lambda v: round(v, 2), b)) for d in dets[:1] for b in d],
                              tracks=kept, covered_frames=sum(1 for b in boxes if b)))
        is_end = p["group"] == "end"
        gpt = None
        if p["sweep"] or is_end:
            mid = frames[int(n * 0.6)]
            gpt = glint_point(mid, [b for bb in boxes for b in bb]) if not is_end else None
            st = p["t0"] + (0.10 if not is_end else 0.05)
            sweep_times.append(st)
        for k in range(n):
            f = frames[k]
            if p["sky"]:
                f = sky_filter(f, 0.5 if is_end else 0.42)
            t = (f0 + k) / FPS
            if p["sweep"] or is_end:
                sw_len = 0.75 if not is_end else 0.9
                u = (t - (p["t0"] + (0.10 if not is_end else 0.05))) / sw_len
                f = light_sweep(f, u, gpt)
            f = put_plates(f, boxes[k])
            if not is_end:
                f = punch(f, k)
            # overlays
            if is_end:
                ea = min(1.0, k / 6)
                f = comp(f, endl, ea)
            else:
                f = comp(f, wm, 1.0)
                g = p["group"]
                if g in labels and group_dur[g] >= 0.85:
                    tg = t - group_start[g]
                    la = min(1.0, tg / 0.2) * min(1.0, (group_start[g] + group_dur[g] - t) / 0.15)
                    f = comp(f, labels[g], max(0, la))
                if 5.2 <= t <= 7.6:
                    ba = min(1.0, (t - 5.2) / 0.2) * min(1.0, (7.6 - t) / 0.2)
                    f = comp(f, badge, ba)
            enc.stdin.write(np.ascontiguousarray(f).tobytes())
            written += 1
            if k == n // 2:
                stills[si] = f
    enc.stdin.close()
    enc.wait()
    LOG["plates"] = plate_log
    LOG["plate_rejects"] = REJECT[:40]
    LOG["raqm"] = RAQM
    LOG["frames"] = written

    # stills for review
    for si, f in stills.items():
        cv2.imwrite(os.path.join(OUT, f"still_{si:02d}.jpg"), cv2.resize(f, (360, 640)), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- audio
    y, _ = librosa.load(os.path.join(IN, "music", minfo["file"]), sr=SR, mono=False, offset=minfo["start"], duration=TOTAL + 0.2)
    if y.ndim == 1:
        y = np.stack([y, y])
    y = y[:, :int(TOTAL * SR)]
    nS = y.shape[1]
    fade = np.ones(nS)
    fi = int(0.6 * SR)
    fade[-fi:] = np.linspace(1, 0, fi) ** 1.5
    fade[:int(0.02 * SR)] = np.linspace(0, 1, int(0.02 * SR))
    music = y * fade * 0.8
    fx = np.zeros(nS)
    place(fx, sfx_thud(), 0.0)
    # duck music under the thud
    duck = np.ones(nS)
    dn = int(0.35 * SR)
    duck[:dn] = 0.45 + 0.55 * np.linspace(0, 1, dn) ** 2
    music = music * duck
    hero = [p for p in plan if p["group"] == "m2"][0]
    place(fx, sfx_whoosh(0.42), max(0, hero["t0"] - 0.36))
    for st in sweep_times:
        place(fx, sfx_shimmer(), st + 0.15)
    place(fx, sfx_cue(), 5.2)
    endp = plan[-1]
    place(fx, sfx_thud() * 0.55, endp["t0"])
    mix = music + fx[None, :] * 0.9
    mix = mix / max(1.0, np.abs(mix).max() / 0.98)
    import soundfile as sf
    sf.write(os.path.join(OUT, "audio.wav"), mix.T, SR)

    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(OUT, "video.mp4"), "-i",
                    os.path.join(OUT, "audio.wav"), "-af", "loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", "48000",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                    os.path.join(OUT, "taleb-pour-detail-demo.mp4")], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(OUT, "taleb-pour-detail-demo.mp4"),
                    "-vf", "fps=1,scale=270:-1,tile=5x2:padding=6:color=black", "-frames:v", "1",
                    os.path.join(OUT, "contact-sheet-1fps.jpg")], check=True)
    with open(os.path.join(OUT, "render-log.json"), "w", encoding="utf-8") as fh:
        json.dump(LOG, fh, indent=1, ensure_ascii=False)
    print(json.dumps(LOG, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
