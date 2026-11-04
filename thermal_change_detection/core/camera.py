"""Lens undistortion. Everything downstream assumes an ideal pinhole camera
with intrinsics K, which is what the homographies need."""
import cv2
import numpy as np


def _as_dist(D):
    return np.zeros(0) if D is None else np.asarray(D, dtype=np.float64).ravel()


class Undistorter:
    def __init__(self, K, D, size, model='plumb_bob'):
        """size = (width, height). The undistorted image keeps the same K.

        ``valid`` is a uint8 mask of pixels that received image content."""
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.size = tuple(int(s) for s in size)
        D = _as_dist(D)
        self._key = (model, D.tobytes())
        w, h = self.size
        if D.size == 0 or not np.any(D):
            self.map1 = self.map2 = None
            self.valid = np.ones((h, w), np.uint8)
            return
        if model == 'equidistant':
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, D[:4], np.eye(3), self.K, self.size, cv2.CV_32FC1)
        else:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, D, None, self.K, self.size, cv2.CV_32FC1)
        self.valid = cv2.remap(np.ones((h, w), np.uint8), self.map1, self.map2,
                               cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    def matches(self, K, D, size, model):
        return (tuple(size) == self.size
                and np.allclose(np.asarray(K, dtype=np.float64).reshape(3, 3), self.K)
                and self._key == (model, _as_dist(D).tobytes()))

    def undistort(self, img):
        if self.map1 is None:
            return img
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
