"""Image alignment: pose-based homography prior refined by pyramidal ECC.

Convention: the warp ``W`` maps *current* pixel coordinates to *reference*
pixel coordinates, i.e. ``ref_warped(x) = ref(W x)`` -- this is what
cv2.findTransformECC estimates with template=current, input=reference, and
what cv2.warpPerspective(..., WARP_INVERSE_MAP) applies.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlignResult:
    ok: bool
    W: np.ndarray            # 3x3, current px -> reference px (refined)
    W0: np.ndarray           # 3x3, same but from the pose prior only
    correlation: float
    correction_px: float     # max corner displacement W vs W0
    overlap: float           # fraction of valid current pixels covered by the reference
    reason: str = ''


def warp_to_current(img, W, size, interp=cv2.INTER_LINEAR):
    """Resample a reference-frame image into the current frame.

    Images use replicated borders so bilinear interpolation does not blend in
    zeros at the edge; masks (INTER_NEAREST) get 0 outside."""
    border = cv2.BORDER_CONSTANT if interp == cv2.INTER_NEAREST else cv2.BORDER_REPLICATE
    return cv2.warpPerspective(img, W, size, flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=border, borderValue=0)


def preprocess(img, valid, use_gradient=False):
    """float32, zero mean / unit std over valid pixels, invalid pixels = 0."""
    f = img.astype(np.float32)
    if use_gradient:
        f = cv2.GaussianBlur(f, (0, 0), 1.5)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        f = cv2.magnitude(gx, gy)
    m = valid > 0
    mu = float(f[m].mean())
    sd = float(f[m].std()) + 1e-6
    out = (f - mu) / sd
    out[~m] = 0.0
    return out


def masked_correlation(a, b, mask):
    m = mask > 0
    if m.sum() < 10:
        return 0.0
    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def _scale_mat(s):
    """Pixel-centre-correct scaling of coordinates for a resize by factor s."""
    o = 0.5 * s - 0.5
    return np.array([[s, 0, o], [0, s, o], [0, 0, 1]], dtype=np.float64)


def _corner_shift(W, W0, w, h):
    c = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float).T
    p, p0 = W @ c, W0 @ c
    p, p0 = p[:2] / p[2], p0[:2] / p0[2]
    return float(np.max(np.linalg.norm(p - p0, axis=0)))


def align(cur, cur_valid, ref, ref_valid, H_cur_ref, cfg) -> AlignResult:
    """Align ``ref`` onto ``cur``. H_cur_ref is the pose prior (ref px -> cur px)."""
    h, w = cur.shape[:2]
    W0 = np.linalg.inv(H_cur_ref)
    W0 /= W0[2, 2]
    W = W0.copy()
    cur_p = preprocess(cur, cur_valid, cfg.use_gradient)
    ref_p = preprocess(ref, ref_valid, cfg.use_gradient)
    ref_m = (ref_valid > 0).astype(np.uint8)

    if cfg.ecc_enabled:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(cfg.max_iterations), float(cfg.eps))
        g = int(cfg.gauss_filt_size)
        g = max(1, g + (1 - g % 2))  # must be odd
        for level in reversed(range(max(1, int(cfg.pyramid_levels)))):
            s = 0.5 ** level
            if level > 0:
                size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
                c_l = cv2.resize(cur_p, size, interpolation=cv2.INTER_AREA)
                r_l = cv2.resize(ref_p, size, interpolation=cv2.INTER_AREA)
                m_l = cv2.resize(ref_m, size, interpolation=cv2.INTER_NEAREST)
            else:
                c_l, r_l, m_l = cur_p, ref_p, ref_m
            S = _scale_mat(s)
            W_l = (S @ W @ np.linalg.inv(S)).astype(np.float32)
            try:
                _, W_l = cv2.findTransformECC(c_l, r_l, W_l, cv2.MOTION_HOMOGRAPHY,
                                              criteria, m_l, g)
                W = np.linalg.inv(S) @ W_l.astype(np.float64) @ S
                W /= W[2, 2]
            except cv2.error:
                pass  # keep the estimate from the previous level

    # Evaluate the final warp at full resolution.
    ref_w = warp_to_current(ref_p, W, (w, h))
    m_w = (warp_to_current(ref_m, W, (w, h), cv2.INTER_NEAREST) > 0) & (cur_valid > 0)
    overlap = float(m_w.sum()) / max(1, int((cur_valid > 0).sum()))
    rho = masked_correlation(cur_p, ref_w, m_w)
    shift = _corner_shift(W, W0, w, h)

    reason = ''
    if overlap < cfg.min_overlap:
        reason = f'overlap {overlap:.2f} < {cfg.min_overlap}'
    elif rho < cfg.min_correlation:
        reason = f'correlation {rho:.2f} < {cfg.min_correlation}'
    elif shift > cfg.max_correction_px:
        reason = f'correction {shift:.0f}px > {cfg.max_correction_px:.0f}px'
    return AlignResult(not reason, W, W0, rho, shift, overlap, reason)
