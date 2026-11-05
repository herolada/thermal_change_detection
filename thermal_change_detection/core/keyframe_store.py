"""Keyframe map: poses in RAM, images as lossless PNGs on disk.

Insertion and retrieval are independent: every processed frame queries the
store for the best reference within (r_match, theta_match), and separately a
frame is stored when no keyframe exists within (r_insert, theta_insert).
"""
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry import axis_angle_deg, optical_axis


@dataclass
class Keyframe:
    id: int
    stamp: float
    T_map_cam: np.ndarray
    K: np.ndarray
    path: str


@dataclass
class Match:
    keyframe: Keyframe
    distance: float
    angle_deg: float
    score: float          # distance + angle_weight_m_per_deg * angle (lower = better)


class KeyframeStore:
    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime('run_%Y%m%d_%H%M%S') + f'_{os.getpid()}'
        self.run_dir = os.path.join(os.path.expanduser(cfg.storage_dir), name)
        os.makedirs(self.run_dir, exist_ok=True)
        self._kfs = []
        self._pos = np.empty((0, 3))
        self._axis = np.empty((0, 3))
        self._stamps = np.empty(0)

    def __len__(self):
        return len(self._kfs)

    def _dist_angle(self, T):
        d = np.linalg.norm(self._pos - T[:3, 3], axis=1)
        a = axis_angle_deg(self._axis, optical_axis(T))
        return d, a

    def query(self, T_map_cam, stamp) -> Optional[Match]:
        if not self._kfs:
            return None
        c = self.cfg
        d, a = self._dist_angle(T_map_cam)
        ok = ((d <= c.r_match) & (a <= c.theta_match_deg)
              & (stamp - self._stamps >= c.min_keyframe_age_s))
        if not ok.any():
            return None
        score = np.where(ok, d + c.angle_weight_m_per_deg * a, np.inf)
        i = int(np.argmin(score))
        return Match(self._kfs[i], float(d[i]), float(a[i]), float(score[i]))

    def needs_insert(self, T_map_cam) -> bool:
        if not self._kfs:
            return True
        d, a = self._dist_angle(T_map_cam)
        return not np.any((d <= self.cfg.r_insert) & (a <= self.cfg.theta_insert_deg))

    def insert(self, image, T_map_cam, K, stamp) -> Keyframe:
        kf_id = len(self._kfs)
        path = os.path.join(self.run_dir, f'kf_{kf_id:06d}.png')
        if not cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise IOError(f'failed to write keyframe {path}')
        kf = Keyframe(kf_id, float(stamp), np.array(T_map_cam, dtype=np.float64),
                      np.array(K, dtype=np.float64), path)
        self._kfs.append(kf)
        self._pos = np.vstack([self._pos, kf.T_map_cam[:3, 3]])
        self._axis = np.vstack([self._axis, optical_axis(kf.T_map_cam)])
        self._stamps = np.append(self._stamps, kf.stamp)
        return kf

    @staticmethod
    def load(kf: Keyframe):
        img = cv2.imread(kf.path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f'failed to read keyframe {kf.path}')
        return img

    def close(self):
        if self.cfg.cleanup_on_exit:
            shutil.rmtree(self.run_dir, ignore_errors=True)
