"""Overlay rendering: grayscale current image + signed diverging heatmap
(red/yellow = warmer, blue/cyan = colder), alpha proportional to |z|,
hatched non-overlap region, blob outlines and a status line."""
from dataclasses import replace

import cv2
import numpy as np

COLORS = {'warmer': (0, 0, 255), 'colder': (255, 80, 0), 'hotspot': (255, 0, 255)}
# On a false-colour heatmap (warm = red/yellow already): white = warmer,
# cyan = colder, green = hotspot.
HEATMAP_COLORS = {'warmer': (255, 255, 255), 'colder': (255, 255, 0), 'hotspot': (0, 255, 0)}


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


def quality_pct(sigma, cfg):
    """Noise sigma -> 0..100 % (100 at ``quality_sigma_good``, 0 at ``quality_sigma_bad``)."""
    good, bad = cfg.quality_sigma_good, cfg.quality_sigma_bad
    return float(100.0 * np.clip((bad - sigma) / max(bad - good, 1e-6), 0.0, 1.0))


def _shown(b, cfg):
    return b.confirmed or cfg.draw_unconfirmed


def _draw_outlines(img, masks, colors, thickness):
    """masks: {kind: bool mask in ``img`` geometry} -> contour lines."""
    for kind, m in masks.items():
        if not m.any():
            continue
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, colors.get(kind, (255, 255, 255)), thickness, cv2.LINE_AA)


def _draw_label(img, b, span, colors):
    """Change blobs: mean difference in % of the scene contrast (p_low..p_high
    grey range), above the blob; hotspots: peak value, below it."""
    x, y, w, h = b.bbox
    color = colors.get(b.kind, (255, 255, 255))
    if b.kind == 'hotspot':
        label, org = f'hot {b.peak_value:.0f}', (x, min(img.shape[0] - 3, y + h + 12))
    else:
        label, org = f'{100.0 * b.mean_value / span:+.0f}%', (x, max(30, y - 4))
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def blob_masks(change, hot_blobs, hot_labels, cfg):
    """{kind: bool mask (undistorted px)} of the blobs that get drawn."""
    masks = {}
    for blobs, labels in ((change.blobs, change.labels) if change else ([], None),
                          (hot_blobs, hot_labels)):
        for kind in {b.kind for b in blobs}:
            lut = np.array([False] + [b.kind == kind and _shown(b, cfg) for b in blobs])
            masks[kind] = masks.get(kind, False) | lut[labels]
    return masks


def render(cur, cur_valid, change, blobs, masks, status, cfg, show_colder=True, und=None,
           raw=None):
    """cur: float32 (undistorted) current image; change: ChangeResult or None;
    blobs: all blobs to draw (change + hotspot); masks: ``blob_masks`` output,
    outlined. Returns (bgr8, (lo, hi)).

    show_colder=False hides negative changes in the heatmap.
    und (Undistorter) + raw (original mono image): draw on ``raw`` instead,
    mapping the change map and the boxes back through the lens model."""
    lo, hi = gray_limits(cur, cur_valid, cfg.gray_low_pct, cfg.gray_high_pct)
    back = und is not None and raw is not None
    gray = to_gray8(raw if back else cur, lo, hi)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    if change is not None:
        valid = change.valid
        z = np.nan_to_num(change.z, nan=0.0)
        if not show_colder:
            z = np.maximum(z, 0.0)
        span = max(cfg.z_max - cfg.z_min, 1e-6)
        alpha = np.clip((np.abs(z) - cfg.z_min) / span, 0.0, 1.0) * cfg.alpha_max * valid
        heat = _heat_colors(z, cfg.z_max)
        if back:
            alpha = und.distort_back(alpha.astype(np.float32))
            heat = und.distort_back(heat)
            valid = und.distort_back(valid.astype(np.uint8), cv2.INTER_NEAREST) > 0
        out = out * (1.0 - alpha[..., None]) + heat * alpha[..., None]

    out = _hatch_invalid(out, valid) if change is not None else out
    return _finish(out, blobs, masks, hi - lo, status, cfg, und if back else None,
                   COLORS), (lo, hi)


def _hatch_invalid(out, valid):
    """Non-overlap: darkened with diagonal hatching."""
    h, w = valid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    hatch = ((xx + yy) % 12) < 2
    inv = ~valid
    out[inv] = out[inv] * 0.45 + 40.0
    out[inv & hatch] = 110.0
    return out


def _finish(out, blobs, masks, span, status, cfg, und, colors):
    """Blob outlines + labels (mapped through the lens model if ``und``), thin
    gray boxes for blobs dropped by the stability check, and the status line.
    span: scene contrast (grey range) the blob percentages refer to."""
    out = np.clip(out, 0, 255).astype(np.uint8)
    if und is not None:
        masks = {k: und.distort_back(m.astype(np.uint8), cv2.INTER_NEAREST) > 0
                 for k, m in masks.items()}
    _draw_outlines(out, masks, colors, cfg.outline_thickness)
    for b in blobs:
        if not _shown(b, cfg):
            continue
        if und is not None:
            b = replace(b, bbox=und.distort_bbox(b.bbox))
        if not b.stable:
            x, y, w, h = b.bbox
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (150, 150, 150), 1)
        elif b.confirmed:
            _draw_label(out, b, max(span, 1e-6), colors)
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, status, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def render_on_heatmap(heatmap, change, blobs, masks, span, status, cfg, und, back):
    """Detections drawn on an external false-colour image (bgr8, original
    camera geometry, same size as the camera image): blob outlines (``masks``,
    undistorted px) in HEATMAP_COLORS, hatched non-overlap. span: scene
    contrast of the camera image (``render``'s hi - lo) for the blob
    percentages. ``back``: stay in the original geometry (viz.distort_overlay),
    else the heatmap is undistorted like the camera image."""
    valid = change.valid if change is not None else None
    if back:
        if valid is not None:
            valid = und.distort_back(valid.astype(np.uint8), cv2.INTER_NEAREST) > 0
        out = heatmap.astype(np.float32)
    else:
        out = und.undistort(heatmap).astype(np.float32)
    if valid is not None:
        out = _hatch_invalid(out, valid)
    return _finish(out, blobs, masks, span, status, cfg, und if back else None,
                   HEATMAP_COLORS)
