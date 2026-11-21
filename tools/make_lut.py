"""House LUT «گالری طلایی» (Golden Gallery) for Taleb Pour Car City.

Clean-luxury grade made for white cars under flat overcast light:
- gentle S-curve: deeper blacks, highlights rolled off so white paint keeps detail
- overall saturation down a little, warm golds / beige leather kept rich
- whites neutral-cool (clean paint), shadows a touch cool
Writes a 33^3 .cube (R fastest), usable by ffmpeg lut3d, Resolve, CapCut, etc.
"""
import sys
import numpy as np

N = 33


def tone(x):
    # black point + soft highlight knee + S-curve
    x = np.clip((x - 0.018) / (1.0 - 0.018), 0, 1)
    s = (np.tanh(2.4 * (x - 0.5)) / np.tanh(1.2) + 1) / 2
    y = 0.55 * s + 0.45 * x
    knee = 0.86
    hi = y > knee
    y = np.where(hi, knee + (1 - knee) * (1 - np.exp(-(y - knee) / (1 - knee) * 1.25)) / (1 - np.exp(-1.25)), y)
    return np.clip(y, 0, 1)


def grade(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # hue (degrees) for gold protection
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    gold_w = np.exp(-(((h - 38) / 22) ** 2)) * np.clip(d * 4, 0, 1)
    sat = 0.86 + 0.24 * gold_w
    out = lum[..., None] + (rgb - lum[..., None]) * sat[..., None]
    # tone per channel
    out = tone(out)
    # neutral-cool whites, slightly cool shadows
    l2 = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    hiw = np.clip((l2 - 0.62) / 0.38, 0, 1)
    low = np.clip((0.35 - l2) / 0.35, 0, 1)
    out[..., 0] -= 0.010 * hiw + 0.008 * low
    out[..., 2] += 0.014 * hiw + 0.012 * low
    return np.clip(out, 0, 1)


def main(path):
    v = np.linspace(0, 1, N)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")  # r fastest when flattened
    rgb = np.stack([r, g, b], -1).reshape(-1, 3)
    out = grade(rgb)
    with open(path, "w") as f:
        f.write('TITLE "Golden Gallery - Taleb Pour Car City"\n')
        f.write(f"LUT_3D_SIZE {N}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for row in out:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "golden-gallery.cube")
