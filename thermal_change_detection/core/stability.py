"""Short-baseline stability check against parallax / occlusion false positives.

A real change (a person, a warm car) is a property of the scene, so its blob
looks about the same when the camera moves a few dozen centimetres. A blob
caused by parallax or changing occlusion (a window behind a closer branch, an
edge slightly misaligned) depends on where exactly the camera is, so it
grows, shrinks or vanishes over such a short baseline.

For a processed frame, an *auxiliary* frame from ``lookback_s`` earlier is
compared with the **same keyframe** (same alignment and change detection),
and its blobs are mapped into the current image through the keyframe:

    cur px --W_cur--> keyframe px --W_aux^-1--> aux px

Blobs are associated by mask overlap (current blob dilated by
``match_radius_px``), same polarity only. Overlapping blobs on both sides are
grouped (connected components of the overlap graph), so a blob that splits
in two or merges with a neighbour is compared as a whole instead of looking
like it disappeared. Per group:

    size_change = 1 - min(A_cur, A_aux) / max(A_cur, A_aux)     in [0, 1]

1 = no counterpart at all. Blobs of groups above ``max_size_change`` are
dropped as unstable. Blobs mostly outside the auxiliary frame's overlap
region (``min_coverage``) cannot be judged and are kept.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .align import AlignResult, warp_to_current
from .change import Blob, ChangeResult


@dataclass
class StabilityResult:
    status: str                              # 'ok' or why the check was skipped
    dt: float = 0.0                          # [s] current - aux stamp
    baseline_m: float = 0.0                  # camera translation between the two frames
    alignment: Optional[AlignResult] = None  # aux frame -> keyframe
    change: Optional[ChangeResult] = None    # aux frame vs. keyframe (aux px)
    aux_labels: Optional[np.ndarray] = None  # aux blob labels mapped into current px
    aux_valid: Optional[np.ndarray] = None   # aux overlap mask mapped into current px
    aux_image: Optional[np.ndarray] = None   # undistorted aux image (debug only)
    cur_blobs: List[Blob] = field(default_factory=list)  # all current blobs (before drop)
    cur_labels: Optional[np.ndarray] = None  # their labels (blob i -> i + 1)
    groups: List[dict] = field(default_factory=list)  # per group: cur, aux, areas, change
    removed: List[Blob] = field(default_factory=list)


class FrameBuffer:
    """Keeps the most recent frames (any payload) for ``max_age_s`` seconds."""

    def __init__(self, max_age_s):
        self.max_age_s = max_age_s
        self._q = deque()

    def push(self, stamp, item):
        self._q.append((stamp, item))
        while self._q and stamp - self._q[0][0] > self.max_age_s:
            self._q.popleft()

    def pick(self, stamp, lookback_s, tol=0.5):
        """(stamp, item) of the buffered frame closest to ``stamp - lookback_s``
        (strictly older than ``stamp``, within ``tol * lookback_s``), else None."""
        best, best_err = None, tol * lookback_s
        for s, item in self._q:
            if s >= stamp:
                continue
            err = abs((stamp - s) - lookback_s)
            if err <= best_err:
                best, best_err = (s, item), err
        return best


def map_aux_to_current(aux_change, W_cur, W_aux, size):
    """Aux labels / overlap mask -> current px through the shared keyframe."""
    M = np.linalg.inv(W_aux) @ W_cur          # current px -> aux px
    M /= M[2, 2]
    lab = warp_to_current(aux_change.labels.astype(np.float32), M, size, cv2.INTER_NEAREST)
    val = warp_to_current(aux_change.valid.astype(np.uint8), M, size, cv2.INTER_NEAREST) > 0
    return np.rint(lab).astype(np.int32), val


def measure(blobs, labels, aux_blobs, aux_labels, aux_valid, cfg):
    """Set ``size_change`` / ``stable`` on ``blobs`` (current px). Returns the groups."""
    n_cur, n_aux = len(blobs), len(aux_blobs)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.match_radius_px + 1,) * 2) \
        if cfg.match_radius_px > 0 else None
    aux_area = np.bincount(aux_labels.ravel(), minlength=n_aux + 1)[1:n_aux + 1]

    parent = list(range(n_cur + n_aux))       # union-find: cur i -> i, aux j -> n_cur + j

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    coverage = np.zeros(n_cur)
    for i, b in enumerate(blobs):
        sel = labels == i + 1
        coverage[i] = float(aux_valid[sel].mean()) if sel.any() else 0.0
        x, y, w, h = b.bbox
        r = cfg.match_radius_px
        y0, y1, x0, x1 = max(0, y - r), y + h + r, max(0, x - r), x + w + r
        roi = sel[y0:y1, x0:x1].astype(np.uint8)
        if k is not None:
            roi = cv2.dilate(roi, k)
        for j in np.unique(aux_labels[y0:y1, x0:x1][roi > 0]):
            if j > 0 and aux_blobs[j - 1].kind == b.kind:
                parent[find(n_cur + j - 1)] = find(i)

    groups = {}
    for i in range(n_cur):
        groups.setdefault(find(i), ([], []))[0].append(i)
    for j in range(n_aux):
        r = find(n_cur + j)
        if r in groups:
            groups[r][1].append(j)

    out = []
    for cur_ids, aux_ids in groups.values():
        a_cur = sum(blobs[i].area for i in cur_ids)
        a_aux = int(sum(aux_area[j] for j in aux_ids))
        cov = sum(coverage[i] * blobs[i].area for i in cur_ids) / max(a_cur, 1)
        change = 1.0 - min(a_cur, a_aux) / max(a_cur, a_aux, 1)
        judged = cov >= cfg.min_coverage
        stable = (not judged) or change <= cfg.max_size_change
        for i in cur_ids:
            blobs[i].size_change = change
            blobs[i].stable = stable
        out.append(dict(cur=cur_ids, aux=aux_ids, area_cur=a_cur, area_aux=a_aux,
                        coverage=cov, size_change=change, judged=judged, stable=stable))
    return out


def drop_unstable(change):
    """Remove unstable blobs from a ChangeResult (relabelling the rest).
    Returns the removed blobs."""
    keep, removed = [], []
    new = np.zeros_like(change.labels)
    for i, b in enumerate(change.blobs):
        if b.stable:
            keep.append(b)
            new[change.labels == i + 1] = len(keep)
        else:
            b.confirmed = False
            removed.append(b)
    change.blobs, change.labels = keep, new
    return removed
