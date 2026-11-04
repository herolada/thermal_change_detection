"""Temporal persistence: a blob is confirmed once it has been seen in N
consecutive processed frames.

The previous frame's blobs are kept as a "count image" (each blob's pixels
hold its count). It is warped into the current frame with the rotation
homography between the two camera poses, dilated by ``radius_px`` to absorb
parallax from the translation, and each current blob inherits
1 + the highest count it overlaps.
"""
import cv2
import numpy as np

from .geometry import rotation_homography


class PersistenceTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._prev = None
        self._prev_stamp = None
        self._prev_T = None
        self._prev_K = None
        if cfg.radius_px > 0:
            d = 2 * cfg.radius_px + 1
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        else:
            self._kernel = None

    def update(self, blobs, labels, stamp, T_map_cam, K):
        h, w = labels.shape
        prev = None
        if self._prev is not None and 0.0 <= stamp - self._prev_stamp <= self.cfg.max_gap_s:
            H = rotation_homography(K, self._prev_K, T_map_cam, self._prev_T)
            prev = cv2.warpPerspective(self._prev, H, (w, h), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if self._kernel is not None:
                prev = cv2.dilate(prev, self._kernel)

        counts = np.zeros((h, w), np.uint8)
        for i, b in enumerate(blobs):
            sel = labels == i + 1
            c = 1 + (int(prev[sel].max()) if prev is not None else 0)
            b.persistence = min(c, 255)
            b.confirmed = c >= self.cfg.required_frames
            counts[sel] = b.persistence

        self._prev, self._prev_stamp = counts, stamp
        self._prev_T, self._prev_K = np.array(T_map_cam), np.array(K)
        return blobs
