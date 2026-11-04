"""Change map between the current image and an aligned reference.

Steps (all on the overlap region only):
  1. robust gain/offset fit  cur ~= a * ref + b   (camera drift, global warming)
  2. misalignment-tolerant signed difference: the current pixel is compared
     with the [min, max] range of the reference in a small neighbourhood
  3. Gaussian smoothing, noise estimate (robust sigma) -> z-score map
  4. hysteresis threshold, morphological opening, connected components -> blobs
Also contains the reference-free absolute hot-spot detector.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

_BIG = np.float32(1e30)


@dataclass
class Blob:
    kind: str                 # 'warmer' | 'colder' | 'hotspot'
    bbox: tuple               # (x, y, w, h) [px]
    centroid: tuple           # (cx, cy) [px]
    area: int                 # [px]
    mean_value: float         # mean signed difference (hotspot: mean intensity)
    peak_value: float         # largest-magnitude difference (hotspot: max intensity)
    peak_z: float             # largest |z| inside the blob (0 for hotspots)
    persistence: int = 1      # consecutive frames this blob has been seen
    confirmed: bool = True


@dataclass
class ChangeResult:
    diff: np.ndarray          # float32 signed difference, image units, NaN outside overlap
    z: np.ndarray             # float32 diff / sigma, NaN outside overlap
    valid: np.ndarray         # bool overlap mask
    ref_adjusted: np.ndarray  # float32 gain/offset-corrected warped reference
    sigma: float
    gain: float
    offset: float
    blobs: List[Blob] = field(default_factory=list)
    labels: Optional[np.ndarray] = None  # int32, blob i has label i + 1


def robust_sigma(v):
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def fit_gain_offset(x, y, mask, n_samples=20000, huber_k=1.5, iters=8, seed=0):
    """Robust (Huber IRLS) fit of y ~= a * x + b over mask."""
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        return 1.0, 0.0
    if idx.size > n_samples:
        idx = np.random.default_rng(seed).choice(idx, n_samples, replace=False)
    xs = x.ravel()[idx].astype(np.float64)
    ys = y.ravel()[idx].astype(np.float64)
    if xs.std() < 1e-6 * (abs(xs.mean()) + 1.0):   # flat scene: offset only
        return 1.0, float(np.median(ys - xs))
    A = np.stack([xs, np.ones_like(xs)], axis=1)
    wts = np.ones_like(xs)
    sol = np.array([1.0, 0.0])
    for _ in range(iters):
        sw = np.sqrt(wts)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], ys * sw, rcond=None)
        r = ys - A @ sol
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        u = np.abs(r) / (huber_k * s)
        wts = np.where(u <= 1.0, 1.0, 1.0 / u)
    a, b = float(sol[0]), float(sol[1])
    if not (0.2 < a < 5.0):                        # implausible -> offset only
        return 1.0, float(np.median(ys - xs))
    return a, b


def neighborhood_diff(cur, ref, valid, radius):
    """Signed difference that is zero wherever ``cur`` lies within the
    [min, max] of ``ref`` over a (2r+1)^2 window, so edges shifted by up to
    ``radius`` px do not show up. Positive = current is warmer."""
    if radius <= 0:
        return cur - ref
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    lo = cv2.erode(np.where(valid, ref, _BIG).astype(np.float32), k)
    hi = cv2.dilate(np.where(valid, ref, -_BIG).astype(np.float32), k)
    return cur - np.clip(cur, lo, hi)


def masked_blur(img, valid, sigma):
    if sigma <= 0:
        return img
    m = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, img, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def hysteresis(strong, weak):
    """Keep connected components of ``weak`` that contain a ``strong`` pixel."""
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(weak, dtype=bool)
    keep = np.unique(lab[strong & weak])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def morph_open(mask, ksize):
    if ksize <= 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k) > 0


def _components(mask, min_area):
    """Yield (pixel mask, bbox, centroid, area) of each connected component."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        yield lab == i, (x, y, w, h), (float(cents[i, 0]), float(cents[i, 1])), area


def compute_change(cur, ref_w, valid, cfg, trace=None) -> ChangeResult:
    """cur, ref_w: float32 images in the current frame; valid: bool overlap mask.

    ``trace``: optional dict, filled with the intermediate maps (for debugging)."""
    valid = valid.astype(bool)
    if cfg.border_erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * cfg.border_erode_px + 1,) * 2)
        valid = cv2.erode(valid.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT,
                          borderValue=0) > 0

    a, b = fit_gain_offset(ref_w, cur, valid, cfg.gain_fit_samples, cfg.huber_k)
    ref_adj = (a * ref_w + b).astype(np.float32)

    # Noise level from the plain (blurred) residual: robust to the changes
    # themselves, and not deflated by the neighbourhood clamping.
    plain = masked_blur(cur - ref_adj, valid, cfg.blur_sigma)
    sigma = max(robust_sigma(plain[valid]), cfg.min_sigma)

    d_raw = neighborhood_diff(cur, ref_adj, valid, cfg.nbhd_radius)
    d = masked_blur(d_raw, valid, cfg.blur_sigma).astype(np.float32)
    z = d / np.float32(sigma)
    if trace is not None:
        trace.update(valid=valid.copy(), plain=plain, nbhd_raw=d_raw, nbhd_blur=d.copy())

    blobs, labels = [], np.zeros(cur.shape[:2], np.int32)
    for sign, kind in ((1.0, 'warmer'), (-1.0, 'colder')):
        zs = np.where(valid, sign * z, 0.0)
        m = hysteresis(zs >= cfg.z_high, zs >= cfg.z_low)
        if trace is not None:
            trace[f'{kind}_weak'] = valid & (zs >= cfg.z_low)
            trace[f'{kind}_strong'] = valid & (zs >= cfg.z_high)
            trace[f'{kind}_hyst'] = m
        m = morph_open(m, cfg.open_kernel)
        if trace is not None:
            trace[f'{kind}_open'] = m
        for sel, bbox, cent, area in _components(m, cfg.min_blob_area):
            vals = d[sel]
            blobs.append(Blob(kind, bbox, cent, area, float(vals.mean()),
                              float(sign * np.max(sign * vals)), float(np.max(zs[sel]))))
            labels[sel] = len(blobs)

    d[~valid] = np.nan
    z[~valid] = np.nan
    return ChangeResult(d, z, valid, ref_adj, sigma, a, b, blobs, labels)


def detect_hotspots(img, valid, cfg):
    """Reference-free detector: pixels above an absolute threshold.
    Returns (blobs, labels)."""
    labels = np.zeros(img.shape[:2], np.int32)
    if cfg.abs_threshold <= 0:
        return [], labels
    m = (valid > 0) & (img >= cfg.abs_threshold)
    m = morph_open(m, cfg.open_kernel)
    blobs = []
    for sel, bbox, cent, area in _components(m, cfg.min_blob_area):
        vals = img[sel]
        blobs.append(Blob('hotspot', bbox, cent, area, float(vals.mean()),
                          float(vals.max()), 0.0))
        labels[sel] = len(blobs)
    return blobs, labels
