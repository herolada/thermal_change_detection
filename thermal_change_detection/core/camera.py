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
        self.D, self.model = D, model
        self._inv_maps = None
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

    def distort_maps(self):
        """remap() maps from undistorted to original (distorted) image
        geometry: for every original pixel, its undistorted coordinates.
        None without distortion. Computed once, then cached."""
        if self.map1 is None:
            return None
        if self._inv_maps is None:
            w, h = self.size
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            pts = np.stack([xx.ravel(), yy.ravel()], axis=1).reshape(-1, 1, 2)
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-6)
            if self.model == 'equidistant':
                und = cv2.fisheye.undistortPoints(pts, self.K, self.D[:4], P=self.K,
                                                  criteria=crit)
            else:
                und = cv2.undistortPointsIter(pts, self.K, self.D, None, self.K, crit)
            und = und.reshape(h, w, 2).astype(np.float32)
            self._inv_maps = (und[..., 0].copy(), und[..., 1].copy())
        return self._inv_maps

    def distort_back(self, img, interpolation=cv2.INTER_LINEAR, border=0):
        """Undistorted image -> original (distorted) image geometry."""
        maps = self.distort_maps()
        if maps is None:
            return img
        return cv2.remap(img, maps[0], maps[1], interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def distort_points(self, pts):
        """(N, 2) undistorted pixel coordinates -> original image coordinates."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.map1 is None or len(pts) == 0:
            return pts
        Ki = np.linalg.inv(self.K)
        n = (np.c_[pts, np.ones(len(pts))] @ Ki.T)[:, :2]
        if self.model == 'equidistant':
            out = cv2.fisheye.distortPoints(n.reshape(-1, 1, 2), self.K, self.D[:4])
        else:
            obj = np.c_[n, np.ones(len(n))].reshape(-1, 1, 3)
            out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.D)
        return out.reshape(-1, 2)

    def distort_bbox(self, bbox, n=8):
        """(x, y, w, h) in the undistorted image -> bounding box of its
        outline in the original image (clipped to it)."""
        x, y, w, h = bbox
        t = np.linspace(0.0, 1.0, n)
        x0, x1, y0, y1 = x, x + w - 1, y, y + h - 1
        edge = np.concatenate([
            np.c_[x0 + t * (x1 - x0), np.full(n, y0)], np.c_[x0 + t * (x1 - x0), np.full(n, y1)],
            np.c_[np.full(n, x0), y0 + t * (y1 - y0)], np.c_[np.full(n, x1), y0 + t * (y1 - y0)]])
        p = self.distort_points(edge)
        W, H = self.size
        lo = np.clip(np.floor(p.min(axis=0)), 0, [W - 1, H - 1]).astype(int)
        hi = np.clip(np.ceil(p.max(axis=0)), 0, [W - 1, H - 1]).astype(int)
        return int(lo[0]), int(lo[1]), int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
