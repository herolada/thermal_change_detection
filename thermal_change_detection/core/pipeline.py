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
from .stability import StabilityResult, drop_unstable, map_aux_to_current, measure
from .viz import render, render_on_heatmap, to_gray8


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
    stability: Optional[StabilityResult] = None  # only with stability.enabled
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

    def _blob_pixels(self, change, hot_blobs, hot_labels):
        """bool mask of the pixels of the blobs that get drawn."""
        show = lambda b: b.confirmed or self.cfg.viz.draw_unconfirmed  # noqa: E731
        mask = np.zeros(hot_labels.shape, bool)
        for blobs, labels in ((change.blobs, change.labels) if change else ([], None),
                              (hot_blobs, hot_labels)):
            if blobs:
                lut = np.array([False] + [show(b) for b in blobs])
                mask |= lut[labels]
        return mask

    def _check_stability(self, aux, stamp, T_map_cam, K, und, kf, ref, ar, ch, debug):
        """Compare the blobs of ``ch`` with those of the auxiliary frame
        (image, stamp, T_map_cam) against the same keyframe; drops unstable blobs."""
        cfg = self.cfg
        a_img, a_stamp, a_T = aux
        st = StabilityResult('ok', dt=stamp - a_stamp,
                             baseline_m=float(np.linalg.norm(T_map_cam[:3, 3] - a_T[:3, 3])))
        a_cur = und.undistort(to_mono(np.asarray(a_img))).astype(np.float32)
        valid = und.valid
        h, w = a_cur.shape[:2]
        H = rotation_homography(K, kf.K, a_T, kf.T_map_cam)
        a_ar = align(a_cur, valid, ref, valid, H, cfg.align,
                     T_map_cur=a_T, T_map_ref=kf.T_map_cam, K_cur=K, K_ref=kf.K)
        st.alignment = a_ar
        if debug:
            st.aux_image = a_cur
        if not a_ar.ok:
            st.status = f'aux frame rejected: {a_ar.reason}'
            return st
        ref_w = warp_to_current(ref, a_ar.W, (w, h))
        ref_valid_w = warp_to_current(valid, a_ar.W, (w, h), cv2.INTER_NEAREST) > 0
        a_ch = compute_change(a_cur, ref_w, ref_valid_w & (valid > 0), cfg.change)
        st.change = a_ch
        st.aux_labels, st.aux_valid = map_aux_to_current(a_ch, ar.W, a_ar.W, (w, h))
        st.cur_blobs, st.cur_labels = list(ch.blobs), ch.labels.copy()
        st.groups = measure(ch.blobs, ch.labels, a_ch.blobs, st.aux_labels, st.aux_valid,
                            cfg.stability)
        st.removed = drop_unstable(ch)
        return st

    def process(self, image, stamp, T_map_cam, K, D=None, distortion_model='plumb_bob',
                debug=False, aux=None, heatmap=None):
        """image: mono8/mono16 (or BGR, converted to gray); stamp in seconds;
        T_map_cam: 4x4 optical-frame camera pose; K: 3x3; D: distortion coeffs.
        aux: optional (image, stamp, T_map_cam) of an earlier frame of the same
        camera, for the stability check (``stability.enabled``).
        heatmap: optional false-colour image (BGR or mono, original camera
        geometry, same aspect ratio) to draw the overlay on instead.
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
            ar = align(cur, valid, ref, valid, H, cfg.align, trace=align_trace,
                       T_map_cur=T_map_cam, T_map_ref=kf.T_map_cam, K_cur=K, K_ref=kf.K)
            if debug:
                res.debug.update(ref=ref, H_prior=H, align_trace=align_trace)
            res.alignment = ar
            head = f'kf{kf.id} d={match.distance:.2f}m a={match.angle_deg:.1f}deg rho={ar.correlation:.2f}'
            if ar.fallback:
                head += f' FALLBACK(corr {ar.correction_px:.0f}>{ar.bound_px:.0f}px)'
            if ar.ok:
                ref_w = warp_to_current(ref, ar.W, (w, h))
                ref_valid_w = warp_to_current(valid, ar.W, (w, h), cv2.INTER_NEAREST) > 0
                change_trace = {} if debug else None
                ch = compute_change(cur, ref_w, ref_valid_w & (valid > 0), cfg.change,
                                    trace=change_trace)
                if debug:
                    res.debug.update(ref_warped=ref_w, change_trace=change_trace)
                if cfg.stability.enabled:
                    if aux is None:
                        res.stability = StabilityResult('no auxiliary frame')
                    elif ch.blobs:
                        res.stability = self._check_stability(
                            aux, stamp, np.asarray(T_map_cam), K, und, kf, ref, ar, ch, debug)
                    else:
                        res.stability = StabilityResult('no blobs')
                self.change_tracker.update(ch.blobs, ch.labels, stamp, T_map_cam, K)
                res.change = ch
                res.status = f'{head} gain={ch.gain:.3f} sigma={ch.sigma:.1f}'
                if res.stability is not None and res.stability.removed:
                    res.status += f' unstable={len(res.stability.removed)}'
            else:
                res.status = f'{head} REJECTED: {ar.reason}'

        if self.store.needs_insert(T_map_cam):
            res.inserted = self.store.insert(cur_raw, T_map_cam, K, stamp)
            res.status += f' +kf{res.inserted.id}'

        res.blobs = (res.change.blobs if res.change else []) + hot_blobs
        drawn = res.blobs + (res.stability.removed if res.stability else [])
        back = cfg.viz.distort_overlay
        res.overlay, (lo, hi) = render(cur, valid, res.change, drawn, res.status, cfg.viz,
                                       show_colder=cfg.change.detect_colder,
                                       und=und if back else None, raw=img if back else None)
        if heatmap is not None:
            hm = np.asarray(heatmap)
            if hm.ndim == 2:
                hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            elif hm.shape[2] == 4:
                hm = cv2.cvtColor(hm, cv2.COLOR_BGRA2BGR)
            if hm.shape[:2] != (h, w):
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_AREA
                                if hm.shape[0] > h else cv2.INTER_LINEAR)
            fill = self._blob_pixels(res.change, hot_blobs, hot_labels)
            res.overlay = render_on_heatmap(hm, res.change, drawn, fill, res.status,
                                            cfg.viz, und, back)
        if res.change is not None:
            ref_view = np.where(res.change.valid, res.change.ref_adjusted, np.nan)
            res.reference_view = to_gray8(ref_view, lo, hi)
        return res
