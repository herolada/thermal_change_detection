"""Image alignment: pose-based homography prior, refined by ECC or by the
plane + parallax model (see plane_parallax.py).

Convention: the warp ``W`` maps *current* pixel coordinates to *reference*
pixel coordinates, i.e. ``ref_warped(x) = ref(W x)`` -- this is what
cv2.findTransformECC estimates with template=current, input=reference, and
what cv2.warpPerspective(..., WARP_INVERSE_MAP) applies.

Both the refinement and the correlation quality gate work on *features*
(``AlignConfig.features``), not necessarily raw intensities: with
``'edges'`` a bright person or car cannot dominate the cost, which is what
made intensity ECC diverge (and the intensity correlation reject good
alignments) on real data.

A refinement that moves image points further from the pose prior than the
physically plausible bound ``disp_base_px + f * |t| / disp_min_depth_m`` is
discarded and the pose prior is used instead (``AlignResult.fallback``).
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .plane_parallax import PlaneParallax

FEATURES = ('intensity', 'gradient', 'edges')
METHODS = ('ecc', 'plane_parallax', 'none')

# 'edges' feature: gradient magnitude, squashed with g / (g + c), c = this
# percentile of the magnitudes, so every clear edge counts about equally.
EDGE_PRE_BLUR = 1.0
EDGE_SATURATION_PCT = 90.0
EDGE_POST_BLUR = 1.5          # widens the basin of convergence


@dataclass
class AlignResult:
    ok: bool
    W: np.ndarray            # 3x3, current px -> reference px (final, after a fallback)
    W0: np.ndarray           # 3x3, same but from the pose prior only
    correlation: float       # gate correlation of the final warp (on the features)
    correction_px: float     # max displacement of the *refined* warp vs. W0
    overlap: float           # fraction of valid current pixels covered by the reference
    reason: str = ''
    bound_px: float = float('inf')   # allowed correction_px
    fallback: bool = False   # refinement exceeded bound_px -> W = W0
    params: Optional[np.ndarray] = None  # plane_parallax: [w (3, rad), m (3, 1/m)]


def warp_to_current(img, W, size, interp=cv2.INTER_LINEAR):
    """Resample a reference-frame image into the current frame.

    Images use replicated borders so bilinear interpolation does not blend in
    zeros at the edge; masks (INTER_NEAREST) get 0 outside."""
    border = cv2.BORDER_CONSTANT if interp == cv2.INTER_NEAREST else cv2.BORDER_REPLICATE
    return cv2.warpPerspective(img, W, size, flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=border, borderValue=0)


def preprocess(img, valid, features='edges'):
    """float32 feature image, zero mean / unit std over valid pixels, 0 elsewhere."""
    if features not in FEATURES:
        raise ValueError(f'align.features must be one of {FEATURES}, got {features!r}')
    f = img.astype(np.float32)
    m = valid > 0
    if features == 'gradient':
        f = cv2.GaussianBlur(f, (0, 0), 1.5)
        f = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
    elif features == 'edges':
        f = cv2.GaussianBlur(f, (0, 0), EDGE_PRE_BLUR)
        g = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
        inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        g[~inner] = 0.0          # the border of the valid region is not an edge
        c = float(np.percentile(g[inner], EDGE_SATURATION_PCT)) + 1e-6 if inner.any() else 1.0
        f = cv2.GaussianBlur(g / (g + c), (0, 0), EDGE_POST_BLUR)
    mu = float(f[m].mean())
    sd = float(f[m].std()) + 1e-6
    out = (f - mu) / sd
    out[~m] = 0.0
    return out.astype(np.float32)


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


def max_displacement(W, W0, valid, step=16):
    """Largest |W x - W0 x| [px] over a grid of valid current pixels (unlike the
    image centre, this also sees shear / perspective corrections)."""
    h, w = valid.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    sel = valid[ys, xs] > 0
    if not sel.any():
        return 0.0
    P = np.stack([xs[sel], ys[sel], np.ones(int(sel.sum()))]).astype(np.float64)
    a, b = W @ P, W0 @ P
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.linalg.norm(a[:2] / a[2] - b[:2] / b[2], axis=0)
    d = np.where(np.isfinite(d), d, np.inf)
    return float(d.max())


def displacement_bound(cfg, K_cur=None, translation_m=0.0):
    """Physically plausible correction of the pose prior [px]: rotation error
    allowance + parallax of the nearest expected scene point."""
    f = float(K_cur[0, 0]) if K_cur is not None else 0.0
    return cfg.disp_base_px + f * translation_m / max(cfg.disp_min_depth_m, 1e-3)


def _refine_ecc(cur_p, ref_p, ref_m, W0, cfg, trace):
    h, w = cur_p.shape[:2]
    W = W0.copy()
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
            err = ''
        except cv2.error as e:
            err = str(e).strip().splitlines()[-1]  # keep the previous level's estimate
        if trace is not None:
            trace.append({'level': level, 'scale': s, 'W': W.copy(), 'error': err})
    return W


def align(cur, cur_valid, ref, ref_valid, H_cur_ref, cfg, trace=None,
          T_map_cur=None, T_map_ref=None, K_cur=None, K_ref=None) -> AlignResult:
    """Align ``ref`` onto ``cur``. H_cur_ref is the pose prior (ref px -> cur px).

    The poses / intrinsics are optional for ``method='ecc'`` (without them the
    displacement bound is just ``disp_base_px``) and required for
    ``'plane_parallax'``.

    ``trace``: optional list; one dict per pyramid level is appended
    (level, scale, W after the level [full-res px], error message or '')."""
    if cfg.method not in METHODS:
        raise ValueError(f'align.method must be one of {METHODS}, got {cfg.method!r}')
    h, w = cur.shape[:2]
    W0 = np.linalg.inv(H_cur_ref)
    W0 /= W0[2, 2]
    cur_p = preprocess(cur, cur_valid, cfg.features)
    ref_p = preprocess(ref, ref_valid, cfg.features)
    ref_m = (ref_valid > 0).astype(np.uint8)
    have_poses = T_map_cur is not None and T_map_ref is not None
    t_norm = (float(np.linalg.norm(np.asarray(T_map_cur)[:3, 3] - np.asarray(T_map_ref)[:3, 3]))
              if have_poses else 0.0)
    bound = displacement_bound(cfg, K_cur, t_norm)

    params = None
    if cfg.method == 'ecc':
        W = _refine_ecc(cur_p, ref_p, ref_m, W0, cfg, trace)
    elif cfg.method == 'plane_parallax':
        if not have_poses or K_cur is None or K_ref is None:
            raise ValueError("align.method='plane_parallax' needs T_map_cur, T_map_ref, "
                             'K_cur and K_ref')
        pp = PlaneParallax(K_cur, K_ref, T_map_cur, T_map_ref, cfg)
        W, params = pp.solve(cur_p, ref_p, cur_valid, ref_valid, trace=trace)
    else:
        W = W0.copy()

    correction = max_displacement(W, W0, cur_valid)
    fallback = not (correction <= bound)
    if fallback:
        W = W0.copy()

    # Evaluate the final warp at full resolution.
    ref_w = warp_to_current(ref_p, W, (w, h))
    m_w = (warp_to_current(ref_m, W, (w, h), cv2.INTER_NEAREST) > 0) & (cur_valid > 0)
    overlap = float(m_w.sum()) / max(1, int((cur_valid > 0).sum()))
    rho = masked_correlation(cur_p, ref_w, m_w)

    reason = ''
    if overlap < cfg.min_overlap:
        reason = f'overlap {overlap:.2f} < {cfg.min_overlap}'
    elif rho < cfg.min_correlation:
        reason = f'correlation {rho:.2f} < {cfg.min_correlation}'
    return AlignResult(not reason, W, W0, rho, correction, overlap, reason, bound, fallback,
                       params)
