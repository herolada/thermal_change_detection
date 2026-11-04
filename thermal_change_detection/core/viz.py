"""Overlay rendering: grayscale current image + signed diverging heatmap
(red/yellow = warmer, blue/cyan = colder), alpha proportional to |z|,
hatched non-overlap region, blob boxes and a status line."""
import cv2
import numpy as np

COLORS = {'warmer': (0, 0, 255), 'colder': (255, 80, 0), 'hotspot': (255, 0, 255)}


def gray_limits(img, valid, lo_pct, hi_pct):
    v = img[valid > 0] if valid is not None else img.ravel()
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return float(lo), float(max(hi, lo + 1e-6))


def to_gray8(img, lo, hi):
    f = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.nan_to_num(f, nan=0.0), 0, 255).astype(np.uint8)


def _heat_colors(z, z_max):
    """Signed z -> BGR float32. warmer: yellow -> red, colder: cyan -> blue."""
    t = np.clip(np.abs(z) / z_max, 0.0, 1.0)
    pos = z >= 0
    col = np.empty(z.shape + (3,), np.float32)
    col[..., 0] = np.where(pos, 0.0, 255.0)
    col[..., 1] = 255.0 * (1.0 - t)
    col[..., 2] = np.where(pos, 255.0, 0.0)
    return col


def _draw_blob(img, b, draw_unconfirmed):
    if not b.confirmed and not draw_unconfirmed:
        return
    x, y, w, h = b.bbox
    color = COLORS.get(b.kind, (255, 255, 255))
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 2 if b.confirmed else 1)
    if b.confirmed:
        if b.kind == 'hotspot':   # label below the box, change labels go above
            label, org = f'hot {b.peak_value:.0f}', (x, min(img.shape[0] - 3, y + h + 12))
        else:
            label, org = f'{b.kind} {b.mean_value:+.0f}', (x, max(30, y - 4))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def render(cur, cur_valid, change, blobs, status, cfg):
    """cur: float32 current image; change: ChangeResult or None;
    blobs: all blobs to draw (change + hotspot). Returns (bgr8, (lo, hi))."""
    lo, hi = gray_limits(cur, cur_valid, cfg.gray_low_pct, cfg.gray_high_pct)
    gray = to_gray8(cur, lo, hi)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    if change is not None:
        valid = change.valid
        z = np.nan_to_num(change.z, nan=0.0)
        span = max(cfg.z_max - cfg.z_min, 1e-6)
        alpha = np.clip((np.abs(z) - cfg.z_min) / span, 0.0, 1.0) * cfg.alpha_max
        alpha = (alpha * valid)[..., None]
        out = out * (1.0 - alpha) + _heat_colors(z, cfg.z_max) * alpha

        # Non-overlap: darkened with diagonal hatching.
        h, w = valid.shape
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((xx + yy) % 12) < 2
        inv = ~valid
        out[inv] = out[inv] * 0.45 + 40.0
        out[inv & hatch] = 110.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    for b in blobs:
        _draw_blob(out, b, cfg.draw_unconfirmed)

    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, status, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out, (lo, hi)
