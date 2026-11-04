"""ROS-independent pipeline: one call per frame.

    pipeline = ChangeDetectionPipeline(PipelineConfig())
    result = pipeline.process(image, stamp_s, T_map_cam, K, D, 'plumb_bob')
"""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .align import AlignResult, align, warp_to_current
from .camera import Undistorter
from .change import Blob, ChangeResult, compute_change, detect_hotspots
from .config import PipelineConfig
from .geometry import rotation_homography
from .keyframe_store import Keyframe, KeyframeStore, Match
from .persistence import PersistenceTracker
from .viz import render, to_gray8


@dataclass
class FrameResult:
    stamp: float
    status: str
    overlay: np.ndarray                         # bgr8
    match: Optional[Match] = None
    inserted: Optional[Keyframe] = None
    alignment: Optional[AlignResult] = None
    change: Optional[ChangeResult] = None
    reference_view: Optional[np.ndarray] = None  # mono8, aligned + gain-corrected reference
    blobs: List[Blob] = field(default_factory=list)  # change + hotspot blobs
    debug: Optional[dict] = None                # intermediates, only with process(debug=True)

    @property
    def detections(self):
        return [b for b in self.blobs if b.confirmed]


def to_mono(img):
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


class ChangeDetectionPipeline:
    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self.store = KeyframeStore(self.cfg.store)
        self.change_tracker = PersistenceTracker(self.cfg.persistence)
        self.hotspot_tracker = PersistenceTracker(self.cfg.persistence)
        self._undist = None

    def close(self):
        self.store.close()

    def _undistorter(self, K, D, size, model):
        if self._undist is None or not self._undist.matches(K, D, size, model):
            self._undist = Undistorter(K, D, size, model)
        return self._undist

    def process(self, image, stamp, T_map_cam, K, D=None, distortion_model='plumb_bob',
                debug=False):
        """image: mono8/mono16 (or BGR, converted to gray); stamp in seconds;
        T_map_cam: 4x4 optical-frame camera pose; K: 3x3; D: distortion coeffs.
        debug=True keeps the intermediate images/warps in ``FrameResult.debug``."""
        cfg = self.cfg
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        img = to_mono(np.asarray(image))
        h, w = img.shape[:2]
        und = self._undistorter(K, D, (w, h), distortion_model)
        cur_raw = und.undistort(img)
        valid = und.valid
        cur = cur_raw.astype(np.float32)

        hot_blobs, hot_labels = detect_hotspots(cur, valid, cfg.hotspot)
        if cfg.hotspot.abs_threshold > 0:
            self.hotspot_tracker.update(hot_blobs, hot_labels, stamp, T_map_cam, K)

        res = FrameResult(stamp, '', None)
        if debug:
            res.debug = {'cur': cur, 'valid': valid, 'K': K, 'T_map_cam': np.array(T_map_cam)}
        match = self.store.query(T_map_cam, stamp)
        if match is None:
            res.status = f'no reference ({len(self.store)} kf)'
        else:
            res.match = match
            kf = match.keyframe
            ref = self.store.load(kf).astype(np.float32)
            H = rotation_homography(K, kf.K, T_map_cam, kf.T_map_cam)
            align_trace = [] if debug else None
            ar = align(cur, valid, ref, valid, H, cfg.align, trace=align_trace)
            if debug:
                res.debug.update(ref=ref, H_prior=H, align_trace=align_trace)
            res.alignment = ar
            head = f'kf{kf.id} d={match.distance:.2f}m a={match.angle_deg:.1f}deg rho={ar.correlation:.2f}'
            if ar.ok:
                ref_w = warp_to_current(ref, ar.W, (w, h))
                ref_valid_w = warp_to_current(valid, ar.W, (w, h), cv2.INTER_NEAREST) > 0
                change_trace = {} if debug else None
                ch = compute_change(cur, ref_w, ref_valid_w & (valid > 0), cfg.change,
                                    trace=change_trace)
                if debug:
                    res.debug.update(ref_warped=ref_w, change_trace=change_trace)
                self.change_tracker.update(ch.blobs, ch.labels, stamp, T_map_cam, K)
                res.change = ch
                res.status = f'{head} gain={ch.gain:.3f} sigma={ch.sigma:.1f}'
            else:
                res.status = f'{head} REJECTED: {ar.reason}'

        if self.store.needs_insert(T_map_cam):
            res.inserted = self.store.insert(cur_raw, T_map_cam, K, stamp)
            res.status += f' +kf{res.inserted.id}'

        res.blobs = (res.change.blobs if res.change else []) + hot_blobs
        res.overlay, (lo, hi) = render(cur, valid, res.change, res.blobs, res.status, cfg.viz)
        if res.change is not None:
            ref_view = np.where(res.change.valid, res.change.ref_adjusted, np.nan)
            res.reference_view = to_gray8(ref_view, lo, hi)
        return res
