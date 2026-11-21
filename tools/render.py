"""Taleb Pour Car City - Reel engine, style «گالری طلایی» (Golden Gallery).

usage: python render.py <work_dir> <template>   template = detail | send | versus

Shared engine (every template inherits every fix):
  music: owned Setavaz tracks; window chosen so the end card lands on a strong beat at 8.5 s
  cuts: on beats, varied lengths (pattern per template)
  shots: ffmpeg decode + house LUT + 1080x1920; speed ramps with frame blending
  look: dark-sky graduated filter, gold light sweep (only on white paint / chrome) + glint
  plates: YOLOv9 detection + IoU tracking + branded plate (edge-clipped plates keep their aspect)
  text: labels on a soft dark band, watermark, follow badge, 1.5 s end card with real contact
  sound: code-made door thud, whoosh, shimmer, badge cue, end impact; loudnorm -14 LUFS
"""
import json
import math
import os
import subprocess
import sys

import cv2
import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beatgrid  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

import arabic_reshaper
from bidi.algorithm import get_display

RAQM = features.check("raqm")
W, H, FPS = 1080, 1920, 30
TOTAL = 10.0  # set from the bar grid of the track in main()
MONTAGE_END = 8.5  # start of the end card = the last downbeat
ROOT = sys.argv[1] if len(sys.argv) > 1 else "work"
TEMPLATE = sys.argv[2] if len(sys.argv) > 2 else "detail"
IN = os.path.join(ROOT, "in")
OUT = os.path.join(ROOT, "out", TEMPLATE)
LUT = os.path.join(ROOT, "golden-gallery.cube")
FONTS = os.path.join(ROOT, "fonts")
os.makedirs(OUT, exist_ok=True)

# public contact from the page bio (instagram.com/shahrak_talebpour, 21 Nov 2026)
PHONE = "013-4213 4343"
HANDLE = "@shahrak_talebpour"
# page setting (owner 21 Nov 2026): real plates stay visible, like the showroom's own page.
# "branded" brings back the tracked plate graphic.
PLATES = os.environ.get("PLATES", "visible")
GOLD = (214, 172, 84)
IVORY = (246, 236, 211)
CREAM = (243, 233, 210)
LOG = {"template": TEMPLATE}


def fa(s):
    return s if RAQM else get_display(arabic_reshaper.reshape(s))


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, f"Vazirmatn-{name}.ttf"), size)


# ============================================================ shots
def S(src, a, b, group=None, label=None, sweep=False, sky=False, ramp=False, cy=0.5, zoom=1.0, px=0.5, py=0.5):
    return dict(src=src, a=a, b=b, group=group, label=label, sweep=sweep, sky=sky, ramp=ramp, cy=cy, zoom=zoom, px=px, py=py)


SHOTS = {
    "c300_grille": S("B", 0.20, 2.85, "c300", "MERCEDES-BENZ  C 300", sweep=True, sky=True, ramp=True, cy=0.55),
    "c300_front_sign": S("B", 2.95, 4.28, "c300", "MERCEDES-BENZ  C 300", sky=True, cy=0.72),
    "c300_rear": S("B", 4.38, 6.15, "c300", "MERCEDES-BENZ  C 300", sky=True, cy=0.52, zoom=1.12, px=0.0),
    "c300_rear_a": S("B", 4.38, 5.25, "c300", "MERCEDES-BENZ  C 300", sky=True, cy=0.52, zoom=1.12, px=0.0),
    "c300_rear_b": S("B", 5.25, 6.15, "c300", "MERCEDES-BENZ  C 300", sky=True, cy=0.52, zoom=1.12, px=0.0),
    "newc_sign": S("C", 0.05, 1.00, "newc", "MERCEDES-BENZ  C-CLASS", sky=True, cy=0.62),
    "newc_hero": S("C", 1.08, 2.10, "newc", "MERCEDES-BENZ  C-CLASS", sweep=True, sky=True, ramp=True, cy=0.62),
    "newc_wheel": S("A", 11.42, 11.76, "newc", None, cy=0.6),
    "eclass_front": S("A", 3.78, 4.36, "e", "MERCEDES-BENZ  E-CLASS", sky=True, cy=0.62),
    "amg_rear": S("A", 9.45, 10.28, "amg", "MERCEDES-AMG", sky=True, cy=0.55),
    "jag": S("A", 2.88, 3.68, "j", "JAGUAR XF", sky=True, cy=0.55),
    "kia_wheel": S("A", 6.62, 7.22, "k", "KIA"),
    "hy_wheel": S("A", 7.32, 7.98, "h"),
    "ty_wheel": S("A", 8.08, 8.65, "t"),
    "g80_sign": S("D", 0.10, 1.85, "g80", "GENESIS G80", sky=True, cy=0.70),
    "g80_front": S("D", 1.95, 2.62, "g80", "GENESIS G80", sky=True, cy=0.55),
    "g80_badge": S("D", 2.72, 3.35, "g80", "GENESIS G80", cy=0.45),
    "g80_light": S("D", 3.45, 4.05, "g80", "GENESIS G80", cy=0.5),
    "g80_rear34": S("D", 4.15, 4.75, "g80", "GENESIS G80", sky=True, cy=0.5),
    "g80_rear": S("D", 4.85, 6.22, "g80", "GENESIS G80", sky=True, cy=0.58),
    "g80_script": S("D", 6.32, 6.95, "g80", "GENESIS G80", cy=0.5),
    "g80_wheel": S("D", 7.05, 7.68, "g80", "GENESIS G80"),
    "g80_seat": S("D", 8.52, 9.15, "g80", "GENESIS G80"),
    "g80_console": S("D", 9.25, 9.88, "g80", "GENESIS G80"),
    "g80_back": S("D", 9.98, 10.55, "g80", "GENESIS G80"),
    "end": S("B", 2.95, 4.28, "end", sky=True),
}


# ============================================================ music + beat grid
# Every Reel is built on the bar grid of its own track (since 21 Nov 2026):
#   - music/showroom.json names the track per template (+ optional drop hint, bpm hint);
#   - tempo and phase come from a comb fit of the kick band around the drop (tools/beatgrid.py), so every cut sits on
#     an exact beat time (librosa beat_track quantises tempo, and generated tracks are not at the tempo they were asked);
#   - the drop = the beat where the low end (< 150 Hz) jumps most (bars after vs bars before), snapped near the hint;
#     downbeats are counted from the drop, so bar lines are musical;
#   - the window starts DROP_BAR bars before the drop and lasts BARS whole bars: the reveal hits the drop, the end
#     card owns the last bar, and the loop restarts on a downbeat.
def load_music_cfg():
    p = os.path.join(IN, "music", "showroom.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def beat_grid(path, bpm_hint=None, drop_hint=None, drop_bar=2, bars=5):
    """bar grid of the track (tools/beatgrid.py: kick-band comb fit around the drop, not librosa's quantised tempo)."""
    y, sr = librosa.load(path, sr=beatgrid.SRA, mono=True)
    rel, info = beatgrid.grid_for(y.astype(np.float64), sr, bpm_hint, drop_hint, drop_bar, bars)
    info = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in info.items()}
    info["file"] = os.path.basename(path)
    if info["contrast"] < 3.5:
        print("WARNING: weak beat grid (contrast %.2f); give a bpm and drop hint in showroom.json" % info["contrast"])
    print("music", info)
    return info, rel


def music_candidates():
    """fallback for tracks without a showroom.json entry: old window search, then the grid around its best drop."""
    files = [f for f in sorted(os.listdir(os.path.join(IN, "music"))) if f.endswith((".mp3", ".flac", ".wav"))]
    return files


# ============================================================ templates (all in beats)
def seg(name, **kw):
    d = dict(SHOTS[name])
    d.update(kind="shot", name=name)
    d.update(kw)
    return d


SEND_LINES = ["بفرست برای رفیقی که", "همیشه میگه", "«یه روز بنز می‌خرم»"]

# Each template: BARS whole bars (the last one is the end card), DROP_BAR = bar where the reveal starts.
# Spec = (segment, beats, transition into it). Beats may be halves (eighth notes). Transitions:
#   cut   hard cut + zoom punch on the beat     whip  motion-blur slide across the beat (+ whoosh ending on it)
#   flash gold-white flash + impact (the drop)
TEMPLATES = {
    "detail": dict(bars=5, drop_bar=2),
    "send": dict(bars=6, drop_bar=1),
    "versus": dict(bars=6, drop_bar=1),
}


def template_specs():
    if TEMPLATE == "detail":
        # «حدس بزن کدوم ماشینه؟»: G80 details on the build without a name, the whole car on the drop.
        pre = [("g80_seat", 2, "cut"), ("g80_console", 2, "cut"), ("g80_wheel", 1, "cut"),
               ("g80_light", 1, "whip"), ("g80_badge", 1, "cut"), ("g80_script", 1, "whip")]
        post = [("g80_sign", 4, "flash"), ("g80_front", 2, "whip"), ("g80_rear34", 1, "cut"), ("g80_rear", 1, "cut")]
        out = [dict(seg=seg(n, label=None, sweep=False), beats=b, tr=t) for n, b, t in pre]
        out[0]["seg"]["hook"] = "حدس بزن کدوم ماشینه؟"
        for k, (n, b, t) in enumerate(post):
            sg = seg(n)
            if k == 0:
                sg.update(sweep=True, ramp=True, reveal="جنسیس G80")
            out.append(dict(seg=sg, beats=b, tr=t))
        return out
    if TEMPLATE == "send":
        card = dict(kind="card", group="card", lines=SEND_LINES, name="card")
        names = [("newc_hero", 4, "flash"), ("newc_wheel", 1, "whip"), ("c300_grille", 3, "cut"), ("amg_rear", 2, "whip"),
                 ("eclass_front", 2, "cut"), ("c300_rear", 2, "whip"), ("newc_sign", 2, "cut")]
        out = [dict(seg=card, beats=4, tr="cut")]
        for k, (n, b, t) in enumerate(names):
            sg = seg(n)
            if k == 0:
                sg.update(sweep=True, ramp=True)
            out.append(dict(seg=sg, beats=b, tr=t))
        return out
    if TEMPLATE == "versus":
        rounds = [
            (None, "c300_front_sign", "g80_sign", "cut", dict(title=True)),
            ("جلو", "c300_grille", "g80_front", "flash", {}),
            ("عقب", "c300_rear_a", "g80_rear", "whip", {}),
            ("سه‌رخ", "c300_rear_b", "g80_rear34", "whip", {}),
            (None, "c300_front_sign", "g80_badge", "flash", dict(ask=True)),
        ]
        out = []
        for r, a, b, t, extra in rounds:
            sg = dict(kind="split", group="vs", top=seg(a), bot=seg(b), round=r, name=f"{a}|{b}")
            sg.update(extra)
            out.append(dict(seg=sg, beats=4, tr=t))
        return out
    raise SystemExit("unknown template " + TEMPLATE)


def allocate(specs, grid, drop_bar):
    """grid = beat times of the window (bars*4 + 1 values, from 0). Cuts land on beats (or exact half beats)."""
    def at(b):
        k = int(math.floor(b))
        if k >= len(grid) - 1:
            return grid[-1]
        return grid[k] + (b - k) * (grid[k + 1] - grid[k])

    plan, b = [], 0.0
    montage_beats = len(grid) - 1 - 4
    total = sum(s["beats"] for s in specs)
    if abs(total - montage_beats) > 1e-6:
        raise SystemExit(f"{TEMPLATE}: specs fill {total} beats, the montage has {montage_beats}")
    for sp in specs:
        p = dict(sp["seg"], t0=at(b), t1=at(b + sp["beats"]), tr=sp["tr"], beat0=b)
        plan.append(p)
        b += sp["beats"]
    reveal = [p for p in plan if abs(p["beat0"] - drop_bar * 4) < 1e-6]
    if not reveal:
        raise SystemExit(f"{TEMPLATE}: no segment starts on the drop (beat {drop_bar * 4})")
    reveal[0]["drop"] = True
    plan.append(dict(SHOTS["end"], kind="end", name="end", t0=grid[-5], t1=grid[-1], tr="cut", beat0=b))
    return plan


# ============================================================ video io
SRC = {"A": "A.mp4", "B": "B.mp4", "C": "C.mp4", "D": "D.mp4"}
_cache = {}


def decode(shot):
    key = (shot["src"], shot["a"], shot["b"], shot.get("zoom", 1.0), shot.get("px", 0.5))
    if key in _cache:
        return _cache[key]
    src = os.path.join(IN, SRC[shot["src"]])
    dur = shot["b"] - shot["a"]
    z = shot.get("zoom", 1.0) if PLATES != "visible" else 1.0
    crop = f"crop=iw/{z}:ih/{z}:(iw-iw/{z})*{shot.get('px', 0.5)}:(ih-ih/{z})*{shot.get('py', 0.5)}," if z > 1.001 else ""
    vf = f"{crop}lut3d={LUT},scale={W}:{H}:flags=lanczos,format=bgr24"
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{shot['a']:.3f}", "-t", f"{dur:.3f}", "-i", src,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    _cache[key] = fr
    return fr


def remap(frames, n_out, ramp):
    """time remap with frame blending (smooth slow parts of ramps at 30 fps)."""
    m = len(frames)
    out = []
    for i in range(n_out):
        u = i / max(1, n_out - 1)
        if ramp:
            p = (1.7 * u - 0.7 * u * u) * (m - 1)  # speed 1.7x -> 0.3x of average
        elif m >= n_out:
            p = (m - n_out) / 2 + i  # real time, centred
        else:
            p = u * (m - 1)  # gentle stretch of a short source
        lo = int(math.floor(p))
        fr = p - lo
        a = frames[min(m - 1, max(0, lo))]
        if fr < 0.08 or lo + 1 >= m:
            out.append(a)
        else:
            b = frames[lo + 1]
            out.append(cv2.addWeighted(a, 1 - fr, b, fr, 0))
    return out


# ============================================================ looks
_ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
_xx, _yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def sky_filter(f, strength=0.42, h=H):
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    g = np.clip(1 - ys / 0.40, 0, 1) ** 1.6
    img = f.astype(np.float32)
    lum = img.mean(-1, keepdims=True) / 255.0
    k = g * strength * np.clip((lum - 0.35) / 0.5, 0, 1)
    tint = np.array([120, 100, 88], np.float32)
    img = img * (1 - k) + tint * k * 0.55
    return np.clip(img, 0, 255).astype(np.uint8)


def light_sweep(f, u, glint_pt=None):
    if u < 0 or u > 1:
        return f
    h, w = f.shape[:2]
    xx, yy = _xx[:h, :w], _yy[:h, :w]
    ang = math.radians(-32)
    d = xx * math.cos(ang) + yy * math.sin(ang)
    dmin, dmax = float(d.min()), float(d.max())
    pos = dmin + (dmax - dmin) * (-0.15 + 1.3 * u)
    band = np.exp(-((d - pos) / (0.06 * (dmax - dmin))) ** 2)
    img = f.astype(np.float32) / 255.0
    lum = img.mean(-1)
    chroma = img.max(-1) - img.min(-1)
    mask = np.clip((lum - 0.40) / 0.45, 0, 1) ** 1.4 * np.clip(1 - chroma * 3.5, 0, 1)
    mask *= np.clip((yy / h - 0.18) / 0.12, 0, 1)  # keep it off the sky band
    a = (band * (0.10 + 0.75 * mask))[..., None] * 0.85
    col = np.array([205, 238, 255], np.float32) / 255.0
    img = 1 - (1 - img) * (1 - a * col)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if glint_pt is not None:
        env = math.exp(-((u - 0.55) / 0.12) ** 2)
        if env > 0.03:
            out = glint(out, glint_pt, env)
    return out


def glint(f, pt, env):
    h, w = f.shape[:2]
    x0, y0 = pt
    r = 150
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    rr = np.sqrt(xs ** 2 + ys ** 2) + 1e-3
    core = np.exp(-(rr / 9) ** 2)
    ray = np.exp(-(np.abs(ys) / 2.2) ** 2) * np.exp(-np.abs(xs) / 60) + np.exp(-(np.abs(xs) / 2.2) ** 2) * np.exp(-np.abs(ys) / 60)
    diag = (np.exp(-(np.abs(xs - ys) / 2.5) ** 2) + np.exp(-(np.abs(xs + ys) / 2.5) ** 2)) * np.exp(-rr / 25) * 0.5
    s = np.clip(core + 0.9 * ray + diag, 0, 1) * env
    x1, y1, x2, y2 = max(0, x0 - r), max(0, y0 - r), min(w, x0 + r + 1), min(h, y0 + r + 1)
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
    h, w = frame.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 6).astype(np.float32)
    m = np.zeros_like(g)
    m[int(h * 0.25):int(h * 0.75), int(w * 0.2):int(w * 0.8)] = 1
    for (x1, y1, x2, y2) in plates:
        m[max(0, y1 - 40):y2 + 40, max(0, x1 - 40):x2 + 40] = 0
    y, x = np.unravel_index(np.argmax(g * m), g.shape)
    return int(x), int(y)


def punch(f, k, n=5, amt=0.045):
    if k >= n:
        return f
    h, w = f.shape[:2]
    s = 1 + amt * (1 - k / n) ** 2
    nw, nh = int(w / s), int(h / s)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(f[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)


# ============================================================ plates
_det = None
REJECT = []


def detector():
    global _det
    if _det is None:
        from open_image_models import create_detector
        _det = create_detector("yolo-v9-s-608-license-plate-end2end")
    return _det


def detect(frames, edges_y=()):
    if PLATES == "visible":
        return [[] for _ in frames]
    res = []
    for f in frames:
        h, w = f.shape[:2]
        boxes = []
        for d in detector().predict(f):
            b = d.bounding_box
            bw, bh = b.x2 - b.x1, b.y2 - b.y1
            ar = bw / max(1, bh)
            edge = b.x1 <= 3 or b.x2 >= w - 3 or b.y2 >= h - 3 or any(abs(b.y1 - e) < 5 or abs(b.y2 - e) < 5 for e in edges_y)
            # angled plates (3/4 views) give squarish boxes; the one false positive seen (a P button) scored <= 0.79
            ok_shape = (1.6 <= ar <= 7.5) or (1.2 <= ar < 1.6 and d.confidence >= 0.70) or (ar >= 0.75 and d.confidence >= 0.88) or (edge and d.confidence >= 0.85)
            if d.confidence < 0.45 or not ok_shape or bw < 24:
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
    tracks = []
    for i, boxes in enumerate(dets):
        for bx in boxes:
            best, bi = None, 0.0
            for t in tracks:
                li = max(t["f"])
                if i - li > 10 or i in t["f"]:
                    continue
                v = iou(t["f"][li], bx)
                if v > bi:
                    best, bi = t, v
            if best is not None and bi > 0.05:
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
        lo2, hi2 = max(0, ks[0] - 6), min(n - 1, ks[-1] + 6)
        full = np.stack([np.interp(np.arange(lo2, hi2 + 1), ks, arr[:, j]) for j in range(4)], 1)
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
    d.text((pw * 0.60 - (bb[2] - bb[0]) / 2, ph / 2 - (bb[3] + bb[1]) / 2 - 6), t, font=f1, fill=(150, 108, 38, 255))
    f2 = font("Medium", 40)
    for k, line in enumerate(["CAR", "CITY"]):
        bb = d.textbbox((0, 0), line, font=f2)
        d.text((150 - (bb[2] - bb[0]) / 2, ph / 2 - 52 + k * 52), line, font=f2, fill=(40, 34, 26, 255))
    d.line([(262, 50), (262, ph - 50)], fill=GOLD + (255,), width=4)
    return np.array(im)


PLATE_ART = None
PLATE_AR = 1040 / 228


_SLOPES = []


def plate_slope(f, box):
    """dominant slope of near-horizontal edges inside the plate box (plate borders, text rows)."""
    x1, y1, x2, y2 = box
    reg = cv2.cvtColor(f[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if reg.size < 100:
        return 0.0
    gx = cv2.Sobel(reg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(reg, cv2.CV_32F, 0, 1, ksize=3)
    sxx, syy, sxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)  # gradient direction
    edge = theta + math.pi / 2  # edge direction
    m = math.tan(edge) if abs(math.cos(edge)) > 0.3 else 0.0
    m = float(np.clip(m, -0.6, 0.6))
    cx = (x1 + x2) / 2
    # temporal smoothing with the slope of the nearest recent plate
    for e in _SLOPES:
        if abs(e[0] - cx) < 90:
            e[1] = 0.8 * e[1] + 0.2 * m
            e[0] = cx
            return e[1]
    _SLOPES.append([cx, m])
    return m


def warp_plate(out, box, src):
    global PLATE_ART
    x1, y1, x2, y2 = box
    h, w = out.shape[:2]
    bw = x2 - x1
    m = plate_slope(src, box)
    cxm, cym = (x1 + x2) / 2, (y1 + y2) / 2
    hw = bw * 0.56
    bh = y2 - y1
    # foreshortened plate: its vertical side is the box height minus the slope rise
    hh = min(bh / 2, max(bw / PLATE_AR * 0.62, (bh - abs(m) * bw) / 2 * 1.08))
    pts = np.float32([[cxm - hw, cym - m * hw - hh], [cxm + hw, cym + m * hw - hh],
                      [cxm + hw, cym + m * hw + hh], [cxm - hw, cym - m * hw + hh]])
    ah, aw = PLATE_ART.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [aw, 0], [aw, ah], [0, ah]]), pts)
    bx1, by1 = int(max(0, pts[:, 0].min() - 2)), int(max(0, pts[:, 1].min() - 2))
    bx2, by2 = int(min(w, pts[:, 0].max() + 2)), int(min(h, pts[:, 1].max() + 2))
    if bx2 <= bx1 or by2 <= by1:
        return out
    T = np.float32([[1, 0, -bx1], [0, 1, -by1], [0, 0, 1]]) @ M
    warped = cv2.warpPerspective(PLATE_ART, T, (bx2 - bx1, by2 - by1), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255.0
    rgb = warped[..., :3][..., ::-1].astype(np.float32)
    reg = out[by1:by2, bx1:bx2].astype(np.float32)
    sc = np.clip(reg.mean() / 170.0, 0.65, 1.05)
    out = out.copy() if out is src else out
    out[by1:by2, bx1:bx2] = np.clip(reg * (1 - a) + rgb * sc * a, 0, 255).astype(np.uint8)
    return out


def put_plates(f, boxes, edges_y=()):
    global PLATE_ART
    if PLATE_ART is None:
        PLATE_ART = make_plate_art()
    h, w = f.shape[:2]
    out = f
    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 4:
            continue
        px, py = int(bw * 0.10) + 3, int(bh * 0.22) + 3
        X1, Y1, X2, Y2 = x1 - px, y1 - py, x2 + px, y2 + py
        edge_box = x1 <= 4 or x2 >= w - 4
        if not edge_box and bw / max(1, bh) < 1.6:
            # angled plate: soften the whole plate area, then a correctly proportioned plate across its width
            bx1, by1, bx2, by2 = max(0, x1 - 2), max(0, y1 - 2), min(w, x2 + 2), min(h, y2 + 2)
            for e in edges_y:
                if by1 < e <= y1:
                    by1 = e
                if y2 <= e < by2:
                    by2 = e
            if bx2 > bx1 + 4 and by2 > by1 + 4:
                if out is f:
                    out = f.copy()
                k = max(11, (max(bx2 - bx1, by2 - by1) // 5) | 1)
                out[by1:by2, bx1:bx2] = cv2.GaussianBlur(out[by1:by2, bx1:bx2], (k, k), 0)
            out = warp_plate(out, (x1, y1, x2, y2), f)
            continue
        # edge-clipped plate: rebuild the full plate size off-frame so the art keeps its aspect
        full_w = int((Y2 - Y1) * PLATE_AR * 0.92)
        if (X2 - X1) < full_w * 0.85:
            if x1 <= 4:
                X1 = X2 - full_w
            elif x2 >= w - 4:
                X2 = X1 + full_w
        tw, th = X2 - X1, Y2 - Y1
        if tw < 4 or th < 4:
            continue
        art = cv2.resize(PLATE_ART, (tw, th), interpolation=cv2.INTER_AREA)
        cx1, cy1, cx2, cy2 = max(0, X1), max(0, Y1), min(w, X2), min(h, Y2)
        for e in edges_y:  # do not spill into the other half of a split screen
            if Y1 < e <= y1:
                cy1 = max(cy1, e)
            if y2 <= e < Y2:
                cy2 = min(cy2, e)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        art = art[cy1 - Y1:cy2 - Y1, cx1 - X1:cx2 - X1]
        a = art[..., 3:4].astype(np.float32) / 255.0
        rgb = art[..., :3][..., ::-1].astype(np.float32)
        reg = out[cy1:cy2, cx1:cx2].astype(np.float32)
        sc = np.clip(reg.mean() / 170.0, 0.65, 1.05)
        if out is f:
            out = f.copy()
        out[cy1:cy2, cx1:cx2] = np.clip(reg * (1 - a) + rgb * sc * a, 0, 255).astype(np.uint8)
    return out


# ============================================================ overlays
def spaced(d, xy, text, fnt, fill, spacing):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += d.textlength(ch, font=fnt) + spacing
    return x


def spaced_len(d, text, fnt, spacing):
    return sum(d.textlength(ch, font=fnt) + spacing for ch in text) - spacing


def soft_band(x0, y0, w, h, alpha=150):
    """dark band fading to the right, for legible small text on white paint."""
    arr = np.zeros((H, W, 4), np.uint8)
    xs = np.clip(1 - (np.arange(W) - x0) / w, 0, 1) ** 1.3
    ys = np.exp(-(((np.arange(H) - (y0 + h / 2)) / (h * 0.7)) ** 2))
    a = (ys[:, None] * xs[None, :] * alpha).astype(np.uint8)
    a[:, :max(0, x0 - 80)] = (a[:, :max(0, x0 - 80)] * 1).astype(np.uint8)
    arr[..., 3] = a
    return Image.fromarray(arr, "RGBA")


def label_layer(text, y0=int(H * 0.165)):
    im = soft_band(0, y0 - 30, 820, 110, 150)
    d = ImageDraw.Draw(im)
    f = font("Light", 42)
    x0 = 84
    d.line([(x0, y0 - 20), (x0 + 70, y0 - 20)], fill=GOLD + (255,), width=3)
    spaced(d, (x0, y0), text, f, IVORY + (255,), 7)
    return np.array(im)


def watermark_layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font("Medium", 26)
    t = "TALEB POUR  CAR CITY"
    tw = spaced_len(d, t, f, 6)
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spaced(ImageDraw.Draw(sh), ((W - tw) / 2 + 2, 130), t, f, (0, 0, 0, 200), 6)
    im = Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(3)), im)
    spaced(ImageDraw.Draw(im), ((W - tw) / 2, 128), t, f, GOLD + (240,), 6)
    return np.array(im)


def badge_layer():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x0, y0, bw, bh = 84, int(H * 0.215), 520, 78
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=39, fill=(14, 12, 10, 185), outline=GOLD + (230,), width=2)
    cx, cy = x0 + 42, y0 + bh // 2
    d.ellipse([cx - 16, cy - 13, cx, cy + 3], fill=GOLD + (255,))
    d.ellipse([cx - 2, cy - 13, cx + 14, cy + 3], fill=GOLD + (255,))
    d.polygon([(cx - 16, cy - 3), (cx + 14, cy - 3), (cx - 1, cy + 16)], fill=GOLD + (255,))
    d.text((x0 + 72, y0 + 20), HANDLE, font=font("Medium", 28), fill=IVORY + (255,))
    fb = font("Bold", 28)
    t = fa("دنبال کنید")
    tw = d.textlength(t, font=fb)
    d.text((x0 + bw - 26 - tw, y0 + 16), t, font=fb, fill=GOLD + (255,))
    return np.array(im)


def endcard_layer():
    grad = np.zeros((H, W, 4), np.uint8)
    grad[..., 3] = (np.clip((_ys[:, 0] - 0.30) / 0.42, 0, 1) ** 1.1 * 225).astype(np.uint8)[:, None]
    im = Image.fromarray(grad, "RGBA")
    d = ImageDraw.Draw(im)
    cy = int(H * 0.58)

    def center(text, fnt, y, fill, rtl=True):
        t = fa(text) if rtl else text
        tw = d.textlength(t, font=fnt)
        d.text(((W - tw) / 2, y), t, font=fnt, fill=fill)

    d.line([(W / 2 - 60, cy - 60), (W / 2 + 60, cy - 60)], fill=GOLD + (255,), width=3)
    center("شهرک اتومبیل طالب‌پور", font("Bold", 80), cy - 30, IVORY + (255,))
    f2 = font("Light", 30)
    e = "TALEB POUR CAR CITY"
    ew = spaced_len(d, e, f2, 9)
    spaced(d, ((W - ew) / 2, cy + 92), e, f2, GOLD + (255,), 9)
    center("بزرگ‌ترین شهرک اتومبیل گیلان", font("Light", 34), cy + 150, IVORY + (230,))
    center("تماس / دایرکت", font("Medium", 50), cy + 220, GOLD + (255,))
    center(PHONE, font("Medium", 46), cy + 296, IVORY + (255,), rtl=False)
    center(HANDLE, font("Light", 34), cy + 362, IVORY + (225,), rtl=False)
    return np.array(im)


def card_layers(lines):
    """text-only hook card: one layer per line (lines appear on beats)."""
    layers = []
    fb = font("Bold", 74)
    y0 = int(H * 0.40)
    for i, line in enumerate(lines):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        t = fa(line)
        tw = d.textlength(t, font=fb)
        col = GOLD if i == len(lines) - 1 else IVORY
        d.text(((W - tw) / 2, y0 + i * 120), t, font=fb, fill=col + (255,))
        layers.append(np.array(im))
    deco = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dd = ImageDraw.Draw(deco)
    dd.line([(W / 2 - 60, y0 - 50), (W / 2 + 60, y0 - 50)], fill=GOLD + (255,), width=3)
    return layers, np.array(deco)


def split_layers():
    """versus: centre divider + model tags + VS mark + round label builder."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H // 2 - 3, W, H // 2 + 3], fill=GOLD + (255,))
    base = np.array(im)

    def tag(text, y):
        t = Image.fromarray(np.zeros((H, W, 4), np.uint8), "RGBA")
        t = Image.alpha_composite(t, soft_band(0, y - 30, 1000, 112, 200))
        dd = ImageDraw.Draw(t)
        spaced(dd, (60, y), text, font("Light", 38), IVORY + (255,), 6)
        return np.array(t)

    top = tag("MERCEDES-BENZ  C 300", 210)
    bot = tag("GENESIS G80", H // 2 + 60)
    return base, top, bot


def pill_layer(text, big=False, y=H // 2):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font("Bold", 64 if big else 46)
    t = fa(text)
    tw = d.textlength(t, font=f)
    ph = 104 if big else 84
    bw = tw + 90
    x0 = (W - bw) / 2
    d.rounded_rectangle([x0, y - ph / 2, x0 + bw, y + ph / 2], radius=ph / 2, fill=(14, 12, 10, 235), outline=GOLD + (255,), width=3)
    bb = d.textbbox((0, 0), t, font=f)
    d.text(((W - tw) / 2, y - (bb[3] + bb[1]) / 2), t, font=f, fill=(GOLD if not big else IVORY) + (255,))
    return np.array(im)


def comp(f, layer, alpha=1.0):
    if alpha <= 0.001:
        return f
    a = layer[..., 3:4].astype(np.float32) / 255.0 * alpha
    rgb = layer[..., :3][..., ::-1].astype(np.float32)
    return np.clip(f.astype(np.float32) * (1 - a) + rgb * a, 0, 255).astype(np.uint8)


# ============================================================ audio
SR = 48000


def lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def sfx_thud():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (52 + 30 * np.exp(-t * 30)) * t) * np.exp(-t * 11)
    panel = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 35) * 0.25
    noise = np.random.default_rng(1).standard_normal(n)
    click = lp(noise, 900) * np.exp(-t * 60) * 3.0
    latch = np.sin(2 * np.pi * 1800 * t) * np.exp(-((t - 0.012) / 0.004) ** 2) * 0.15
    s = body * 0.9 + panel + click + latch
    return s / np.abs(s).max() * 0.9


def sfx_impact():
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * (44 + 20 * np.exp(-t * 8)) * t) * np.exp(-t * 3.2)
    s += lp(np.random.default_rng(3).standard_normal(n), 400) * np.exp(-t * 14) * 2
    return s / np.abs(s).max() * 0.8


def sfx_whoosh(dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.random.default_rng(2).standard_normal(n)
    env = np.sin(np.pi * (t / dur) ** 1.6) ** 2
    out = np.zeros(n)
    y1 = 0.0
    for i in range(n):
        a = np.exp(-2 * np.pi * (300 + 3500 * (t[i] / dur) ** 2) / SR)
        y1 = (1 - a) * x[i] + a * y1
        out[i] = y1
    out *= env
    return out / np.abs(out).max() * 0.55


def sfx_shimmer(dur=0.7):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for k, fr in enumerate([2637, 3136, 3951, 5274]):
        s += np.sin(2 * np.pi * fr * t + k) * np.exp(-((t - 0.12 + 0.04 * k) / 0.18) ** 2)
    return s * 0.12


def sfx_cue():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1318.5 * t) * (t < 0.12) + np.sin(2 * np.pi * 1760 * t) * (t >= 0.09)) * np.exp(-t * 9) * 0.18


def sfx_tick():
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 80) * 0.25


def place(buf, s, t):
    i = int(t * SR)
    j = min(len(buf), i + len(s))
    if j > i >= 0:
        buf[i:j] += s[:j - i]


# ============================================================ render
def render_shot_frames(shot, n):
    frames = remap(list(decode(shot)), n, shot.get("ramp", False))
    return frames


def crop_half(f, cy):
    hh = H // 2
    c = int(cy * H)
    y0 = int(np.clip(c - hh // 2, 0, H - hh))
    return f[y0:y0 + hh]


def whip(f, u, direction=1):
    """motion-blur slide: u in [0, 1] = strength (1 = at the cut)."""
    if u <= 0.02:
        return f
    sh = int(direction * u * W * 0.22)
    M = np.float32([[1, 0, sh], [0, 1, 0]])
    g = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    L = max(3, int(90 * u)) | 1
    return cv2.blur(g, (L, 1))


def flash(f, k, n=5):
    if k >= n:
        return f
    a = 0.85 * (1 - k / n) ** 1.5
    col = np.array([214, 238, 255], np.float32)  # warm white (BGR)
    return np.clip(f.astype(np.float32) * (1 - a) + col * a, 0, 255).astype(np.uint8)


def lowpass_close(x, u_from=0.0, u_to=0.85, fc=1200):
    """the last bar closes like a filter (dry -> low passed), so the loop back to the build sounds like a turnaround."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, fc, btype="low", fs=SR, output="sos")
    lpx = sosfiltfilt(sos, x, axis=-1)
    u = np.linspace(u_from, u_to, x.shape[-1]) ** 1.3
    return x * (1 - u) + lpx * u


def main():
    global TOTAL, MONTAGE_END
    cfg = load_music_cfg()
    tcfg = dict(TEMPLATES[TEMPLATE])
    mc = cfg.get(TEMPLATE, {})
    files = music_candidates()
    mfile = mc.get("file") or files[{"detail": 0, "send": 1, "versus": 2}.get(TEMPLATE, 0) % len(files)]
    minfo, grid = beat_grid(os.path.join(IN, "music", mfile), mc.get("bpm"), mc.get("drop"),
                            tcfg["drop_bar"], tcfg["bars"])
    minfo["title"] = mc.get("title")
    TOTAL = grid[-1]
    MONTAGE_END = grid[-5]
    LOG["music"] = minfo
    plan = allocate(template_specs(), grid, tcfg["drop_bar"])
    drop_t = grid[tcfg["drop_bar"] * 4]
    beat_frames = sorted(set(int(round(g * FPS)) for g in grid[:-1]))
    LOG["cuts"] = [dict(kind=p.get("kind"), name=p.get("name"), tr=p.get("tr"), beat=p["beat0"], t0=round(p["t0"], 3),
                        t1=round(p["t1"], 3), drop=bool(p.get("drop"))) for p in plan]
    # sync check of the plan itself: every cut vs the nearest beat / eighth, after frame rounding
    eighths = sorted(set([g for g in grid] + [(a + b) / 2 for a, b in zip(grid[:-1], grid[1:])]))
    errs = []
    for p in plan:
        tf = round(p["t0"] * FPS) / FPS
        errs.append(round(1000 * min(abs(tf - e) for e in eighths), 1))
    LOG["sync_plan"] = dict(max_err_ms=max(errs), mean_err_ms=round(float(np.mean(errs)), 1), frame_ms=round(1000 / FPS, 1),
                            drop_t=round(drop_t, 3), drop_cut=[p["name"] for p in plan if p.get("drop")])
    print(json.dumps(LOG, indent=1, ensure_ascii=False))

    wm = watermark_layer()
    badge = badge_layer()
    endl = endcard_layer()
    bar = grid[4] - grid[0]
    badge_t0 = grid[-9] if TEMPLATE != "versus" else None  # one bar before the end card
    badge_t1 = grid[-5] - 0.1 if badge_t0 is not None else None
    labels = {}
    gstart, gdur = {}, {}
    run, prev = -1, None
    for p in plan:
        g = p.get("group") if p.get("label") or p.get("kind") != "shot" else ("nolabel", p["name"])
        if g != prev:
            run += 1
            prev = g
        p["run"] = run
        gdur[run] = gdur.get(run, 0) + (p["t1"] - p["t0"])
        if run not in gstart:
            gstart[run] = p["t0"]
        if run not in labels and p.get("label"):
            labels[run] = label_layer(p["label"])
        if p.get("group") not in gstart:
            gstart[p.get("group")] = p["t0"]
    split_base, tag_top, tag_bot = split_layers()
    hook = next((p.get("hook") for p in plan if p.get("hook")), None)
    hook_l = pill_layer(hook, big=True, y=int(H * 0.30)) if hook else None

    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "17", "-pix_fmt", "yuv420p", os.path.join(OUT, "video.mp4")], stdin=subprocess.PIPE)
    total_frames = int(round(TOTAL * FPS))
    plate_log, sweeps, stills, events = [], [], {}, []
    first_frame = None
    for si, p in enumerate(plan):
        f0 = int(round(p["t0"] * FPS))
        f1 = total_frames if si == len(plan) - 1 else int(round(p["t1"] * FPS))
        n = f1 - f0
        kind = p.get("kind")
        nxt_tr = plan[si + 1]["tr"] if si + 1 < len(plan) else None
        edges = ()
        if kind == "card":
            layers, deco = card_layers(p["lines"])
            nxt = plan[si + 1] if si + 1 < len(plan) else SHOTS["c300_grille"]
            bgsrc = decode(nxt)
            line_t = [grid[k] for k in range(len(layers))]  # one line per beat
            frames = []
            for k in range(n):
                t = (f0 + k) / FPS
                bf = bgsrc[min(len(bgsrc) - 1, k // 3)]
                bf = cv2.GaussianBlur(cv2.resize(bf, (W // 4, H // 4)), (0, 0), 6)
                bf = cv2.resize(bf, (W, H), interpolation=cv2.INTER_LINEAR)
                f = (bf.astype(np.float32) * 0.28 + np.array([12, 10, 10], np.float32)).clip(0, 255).astype(np.uint8)
                f = comp(f, wm)
                f = comp(f, deco)
                for li, L in enumerate(layers):
                    if t >= line_t[li] - 1e-3:
                        f = comp(f, L, min(1.0, (t - line_t[li]) * FPS / 3 + 0.34))
                frames.append(f)
            for lt in line_t:
                events.append(("tick", lt))
            boxes = [[] for _ in range(n)]
        elif kind == "split":
            top = render_shot_frames(p["top"], n)
            bot = render_shot_frames(p["bot"], n)
            frames = []
            for k in range(n):
                a = crop_half(top[k], p["top"]["cy"])
                b = crop_half(bot[k], p["bot"]["cy"])
                if p["top"].get("sky"):
                    a = sky_filter(a, 0.40, a.shape[0])
                if p["bot"].get("sky"):
                    b = sky_filter(b, 0.40, b.shape[0])
                frames.append(np.ascontiguousarray(np.vstack([a, b])))
            edges = (H // 2,)
            dets = detect(frames, edges)
            boxes, kept = track(dets, n)
        else:
            frames = render_shot_frames(p, n)
            dets = detect(frames)
            boxes, kept = track(dets, n)
        if kind != "card":
            plate_log.append(dict(seg=si, name=p.get("name", kind), frames=n, detected=sum(1 for d in boxes if d)))
        is_end = kind == "end"
        sweep = (p.get("sweep") and kind == "shot") or is_end
        gpt = None
        st = p["t0"] + (0.10 if not is_end else 0.05)
        if sweep:
            sweeps.append(st)
            if not is_end:
                gpt = glint_point(frames[int(n * 0.6)], [b for bb in boxes for b in bb])
        rl_title = pill_layer("این یا این؟", big=True) if (kind == "split" and p.get("title")) else None
        if kind == "split" and p.get("round"):
            rl = pill_layer(p["round"])
        elif kind == "split" and p.get("ask"):
            rl = pill_layer("تو کدومو برمی‌داری؟", big=True)
        else:
            rl = None
        reveal_l = pill_layer(p["reveal"], big=True, y=int(H * 0.30)) if p.get("reveal") else None
        # transition sounds: whoosh peaks on the cut, impact on flashes
        if si > 0 and p["tr"] == "whip":
            events.append(("whoosh", p["t0"]))
        if p["tr"] == "flash":
            events.append(("impact", p["t0"]))
        if p.get("drop"):
            events.append(("riser", p["t0"]))
        for k in range(n):
            f = frames[k]
            fi = f0 + k
            t = fi / FPS
            if kind == "shot" and p.get("sky"):
                f = sky_filter(f, 0.42)
            if is_end:
                f = sky_filter(f, 0.5)
            if sweep:
                f = light_sweep(f, (t - st) / (0.9 if is_end else 0.75), gpt)
            f = put_plates(f, boxes[k], edges)
            if kind in ("shot", "split"):
                if k < 5:
                    f = punch(f, k, amt=0.09 if p["tr"] == "flash" else 0.045)
                else:  # a small pulse on every beat inside the shot
                    lb = max(b for b in beat_frames if b <= fi)
                    if lb > f0:
                        f = punch(f, fi - lb, n=4, amt=0.018)
            if is_end:
                f = comp(f, endl, min(1.0, k / 6))
            elif kind == "split":
                f = comp(f, split_base)
                f = comp(f, tag_top, min(1.0, (t - gstart["vs"]) / 0.2))
                f = comp(f, tag_bot, min(1.0, (t - gstart["vs"]) / 0.2))
                if rl_title is not None:
                    f = comp(f, rl_title, 1.0 if k < n - 2 else 0.5)
                elif rl is not None:
                    f = comp(f, rl, min(1.0, (k + 2) / 4))
                f = comp(f, wm)
            elif kind == "shot":
                f = comp(f, wm)
                g = p["run"]
                if g in labels and gdur[g] >= 0.75:
                    tg = t - gstart[g]
                    la = min(1.0, tg / 0.2) * min(1.0, (gstart[g] + gdur[g] - t) / 0.15)
                    f = comp(f, labels[g], max(0, la))
                if hook_l is not None and t < drop_t:
                    f = comp(f, hook_l, min(1.0, (t + 0.1) / 0.1) * min(1.0, (drop_t - t) / 0.08))
                if reveal_l is not None and t >= p["t0"] + bar / 4 - 1e-3:
                    f = comp(f, reveal_l, min(1.0, (t - p["t0"] - bar / 4) / 0.12 + 0.3))
                if badge_t0 is not None and badge_t0 <= t <= badge_t1:
                    f = comp(f, badge, min(1.0, (t - badge_t0) / 0.2) * min(1.0, (badge_t1 - t) / 0.2))
            # transitions across the cut
            if p["tr"] == "whip" and si > 0 and k < 3:
                f = whip(f, 1 - k / 3, direction=1)
            if nxt_tr == "whip" and k >= n - 3:
                f = whip(f, (k - (n - 3) + 1) / 3, direction=-1)
            if p["tr"] == "flash":
                f = flash(f, k)
            # loop: the end card's last frames dissolve into the opening frame
            if is_end and first_frame is not None and k >= n - 6:
                u = (k - (n - 6) + 1) / 6
                f = cv2.addWeighted(f, 1 - u, first_frame, u, 0)
            if fi == 0:
                first_frame = f.copy()
            enc.stdin.write(np.ascontiguousarray(f).tobytes())
            if k == n // 2:
                stills[si] = f
    enc.stdin.close()
    enc.wait()
    LOG["plates"] = plate_log
    LOG["plate_rejects"] = REJECT[:30]
    LOG["raqm"] = RAQM
    LOG["plates_mode"] = PLATES
    for si, f in stills.items():
        cv2.imwrite(os.path.join(OUT, f"still_{si:02d}.jpg"), cv2.resize(f, (360, 640)), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- audio: exactly the window of whole bars; the last bar closes like a filter, then the loop restarts
    y, _ = librosa.load(os.path.join(IN, "music", mfile), sr=SR, mono=False, offset=minfo["start"], duration=TOTAL + 0.2)
    if y.ndim == 1:
        y = np.stack([y, y])
    nS = int(round(TOTAL * SR))
    y = y[:, :nS]
    if y.shape[1] < nS:
        y = np.pad(y, ((0, 0), (0, nS - y.shape[1])))
    eb = int(round(MONTAGE_END * SR))
    y[:, eb:] = lowpass_close(y[:, eb:])
    fade = np.ones(nS)
    fade[:int(0.005 * SR)] = np.linspace(0, 1, int(0.005 * SR))
    fade[-int(0.03 * SR):] = np.linspace(1, 0, int(0.03 * SR))
    music = y * fade * 0.8
    fx = np.zeros(nS)
    if plan[0].get("kind") == "card":
        music[:, :int(drop_t * SR)] *= 0.7
    else:
        place(fx, sfx_thud() * 0.8, 0.0)
    wd = 0.42
    for ev, t in events:
        if ev == "tick":
            place(fx, sfx_tick(), t)
        elif ev == "whoosh":
            place(fx, sfx_whoosh(wd) * 0.6, max(0, t - 0.65 * wd))
        elif ev == "impact":
            place(fx, sfx_impact() * 0.7, t)
        elif ev == "riser":
            place(fx, sfx_whoosh(0.8) * 0.8, max(0, t - 0.8 * 0.8))
            place(fx, sfx_thud() * 0.6, t)
    for st in sweeps:
        place(fx, sfx_shimmer(), st + 0.15)
    if badge_t0 is not None:
        place(fx, sfx_cue(), badge_t0)
    place(fx, sfx_impact() * 0.8, MONTAGE_END)
    place(fx, sfx_thud() * 0.45, MONTAGE_END)
    fx[-int(0.03 * SR):] *= np.linspace(1, 0, int(0.03 * SR))
    mix = music + fx[None, :] * 0.9
    mix = mix / max(1.0, np.abs(mix).max() / 0.98)
    sf.write(os.path.join(OUT, "audio.wav"), mix.T, SR)

    final = os.path.join(OUT, f"taleb-pour-{TEMPLATE}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(OUT, "video.mp4"), "-i",
                    os.path.join(OUT, "audio.wav"), "-af", "loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", "48000",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", final, "-vf",
                    "fps=1,scale=270:-1,tile=6x2:padding=6:color=black", "-frames:v", "1",
                    os.path.join(OUT, f"contact-sheet-{TEMPLATE}-1fps.jpg")], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", final, "-vf",
                    f"fps={10 / TOTAL:.5f},scale=270:-1,tile=5x2:padding=6:color=black", "-frames:v", "1",
                    os.path.join(OUT, f"frames10-{TEMPLATE}.jpg")], check=True)
    os.remove(os.path.join(OUT, "video.mp4"))
    os.remove(os.path.join(OUT, "audio.wav"))
    with open(os.path.join(OUT, f"render-log-{TEMPLATE}.json"), "w", encoding="utf-8") as fh:
        json.dump(LOG, fh, indent=1, ensure_ascii=False)
    print(json.dumps(LOG, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
