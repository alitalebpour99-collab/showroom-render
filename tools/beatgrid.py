"""Beat grid for the showroom Reels (shared by render.py and sync_check.py).

Why not librosa.beat.beat_track: its tempo is quantised to onset-frame bins (e.g. 123.0 / 129.2 BPM at hop 512),
and AI-made tracks are not always at the tempo they were asked for (Zarrin was asked for 120, plays at ~126).
A 3 % tempo error moves the last cut of a 10 s Reel by a whole eighth note.

Here: kick-band envelope at 2.9 ms resolution, then a comb search: for every tempo in the range and every phase,
sum the kick onsets that fall on the grid; the true grid keeps the kick on it for the whole window. Fitted LOCALLY
(around the drop), because generated tracks can drift a little over minutes.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

SRA = 22050
HOP = 64  # samples -> 2.902 ms


def kick_onsets(y, sr=SRA):
    sos = butter(4, 150, btype="low", fs=sr, output="sos")
    lo = sosfiltfilt(sos, y)
    n = len(lo) // HOP
    e = np.sqrt((lo[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)
    on = np.zeros_like(e)
    on[4:] = np.maximum(0, e[4:] - e[:-4])
    return on, HOP / sr


def comb(on, hop_s, t0, t1, bpm_lo, bpm_hi, step=0.01, ph_step=None):
    """best (bpm, phase, contrast); phase = absolute time of a beat (seconds)."""
    t0, t1 = max(0.0, t0), min(t1, len(on) * hop_s - 0.01)
    ph_step = ph_step or 2 * hop_s
    best = (0.0, 60.0 / bpm_lo, 0.0, 0.0)
    for bpm in np.arange(bpm_lo, bpm_hi, step):
        p = 60.0 / bpm
        ph = np.arange(0, p, ph_step)
        k = np.arange(int((t1 - t0) / p))
        T = t0 + ph[:, None] + k[None, :] * p
        s = on[np.clip(np.round(T / hop_s).astype(int), 0, len(on) - 1)].sum(1)
        c = float(s.max() / (s.mean() + 1e-12))
        if c > best[0]:
            best = (c, p, float(t0 + ph[int(s.argmax())]), bpm)
    c, p, phase, bpm = best
    return float(bpm), p, phase, c


def fit(y, sr=SRA, bpm_hint=None, center=None, span=(-14.0, 18.0)):
    """tempo + phase near `center` (or over 10..100 s). Returns (period, phase, bpm, contrast)."""
    on, hop_s = kick_onsets(y, sr)
    dur = len(on) * hop_s
    lo, hi = (bpm_hint * 0.92, bpm_hint * 1.08) if bpm_hint else (95.0, 145.0)
    if center is None:  # blind: several 30 s windows, keep the clearest (a breakdown without kick can fool one long window)
        wins = [(a, a + 30.0) for a in (8.0, 30.0, 52.0, 74.0) if a + 30.0 < dur - 2] or [(0.0, dur - 1)]
        res = [comb(on, hop_s, a, b, lo, hi, step=0.02) + (a, b) for a, b in wins]
        bpm, p, phase, c, t0, t1 = max(res, key=lambda r: r[3])
    else:
        t0, t1 = center + span[0], center + span[1]
        bpm, p, phase, c = comb(on, hop_s, t0, t1, lo, hi, step=0.02)
    # refine around the winner
    bpm, p, phase, c = comb(on, hop_s, t0, t1, bpm - 0.05, bpm + 0.05, step=0.002, ph_step=hop_s)
    return p, phase, bpm, c


def kick_level(y, sr=SRA):
    sos = butter(4, 150, btype="low", fs=sr, output="sos")
    lo = sosfiltfilt(sos, y)
    n = len(lo) // HOP
    return np.sqrt((lo[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12), HOP / sr


def find_drop(y, sr, period, phase, drop_bar, bars, hint=None, tmax=75.0):
    """the beat where the low end jumps most (2 bars after vs 2 bars before), with a hard hit on the beat itself;
    near the hint when there is one. Returns (time, ratio)."""
    e, hop_s = kick_level(y, sr)
    dur = len(e) * hop_s
    bar = 4 * period

    def span(t0, t1):
        a, b = int(max(0, t0) / hop_s), int(min(dur, t1) / hop_s)
        return float(e[a:b].mean()) if b > a else 0.0

    best = None
    k0 = int(np.ceil((drop_bar * bar - phase) / period))
    t = phase + k0 * period
    while t < min(tmax, dur - (bars - drop_bar) * bar - 0.3):
        r = (span(t, t + 2 * bar) + 1e-9) / (span(t - 2 * bar, t) + 1e-9)
        r *= 1.0 + 0.5 * (span(t, t + 0.12) / (span(t - 0.25, t - 0.03) + 1e-9) > 1.5)
        if hint is not None:
            r *= float(np.exp(-((t - hint) / (1.5 * bar)) ** 2))
        if best is None or r > best[1]:
            best = (float(t), float(r))
        t += period
    return best


def grid_for(y, sr, bpm_hint=None, drop_hint=None, drop_bar=2, bars=5):
    """full answer for a Reel: (beat times of the window from 0, info)."""
    p, ph, bpm, c = fit(y, sr, bpm_hint, center=drop_hint)
    drop, ratio = find_drop(y, sr, p, ph, drop_bar, bars, drop_hint, tmax=75.0 if drop_hint is not None else 45.0)
    # refit right around the drop (drift), then snap the drop to the refined grid
    p, ph, bpm, c = fit(y, sr, bpm or bpm_hint, center=drop)
    k = round((drop - ph) / p)
    drop = ph + k * p
    start = drop - drop_bar * 4 * p
    rel = [float(i * p) for i in range(bars * 4 + 1)]
    return rel, dict(bpm=round(bpm, 3), period=round(p, 5), contrast=round(c, 2), drop=round(drop, 3),
                     drop_ratio=round(ratio, 2), start=round(start, 4), bars=bars, drop_bar=drop_bar, length=round(rel[-1], 3))
