"""Plane + parallax refinement of the pose prior.

Instead of letting ECC pick any of the 8 homography parameters, the warp is
built from the physics of the two camera poses:

    W(p) = K_ref  dR(w)  (R_rc + t_rc m^T)  K_cur^-1         (current px -> reference px)

* ``R_rc, t_rc``: rotation / translation from the current to the reference
  camera, known from localisation (T_ref^-1 T_cur).
* ``m = n / d`` (3 params, [1/m]): the scene plane ``n^T X = d`` in the current
  camera. For a plane facing the camera at depth Z, ``m = (0, 0, 1/Z)``.
  ``(R_rc + t_rc m^T)`` is the exact plane-induced homography: every point
  of that plane lands where it should, including its parallax. Because ``t``
  is fixed, parallax can only follow the epipolar geometry of the real
  translation; if the robot hardly moved (t ~ 0), ``m`` has no effect.
* ``w`` (3 params, rad): a small rotation correction (heading / extrinsic
  calibration / image-TF timing error), applied in the reference camera.

The 6 parameters are estimated by robust Gauss-Newton on the photometric
residual ``cur(x) - (a ref(W(p) x) + b)`` of the feature images (gain ``a``
and bias ``b`` are estimated jointly), coarse to fine over the pyramid:

* Huber IRLS weights: pixels that disagree strongly (a person in only one of
  the images) get down-weighted instead of pulling the solution.
* Gaussian priors on ``w`` (around 0) and ``m`` (around a plane
  ``plane_depth_m`` ahead) keep the solution physical when the image carries
  little information (e.g. ``m`` with almost no translation).

The Jacobian ``d ref(W(p) x) / dp`` is the image gradient of the reference at
the warped point times ``dx_ref/dp``; the latter is obtained by finite
differences of the 3x3 warp, which is cheap.
"""
import cv2
import numpy as np

HUBER_K = 1.5


def _scale_mat(s):
    o = 0.5 * s - 0.5
    return np.array([[s, 0, o], [0, s, o], [0, 0, 1]], dtype=np.float64)


def rodrigues(w):
    R, _ = cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))
    return R


def sample(img, u, v, interp=cv2.INTER_LINEAR):
    """``img`` at float coordinates (u, v) (1-D arrays). cv2.remap needs maps
    narrower than 32767, so the samples are laid out in 1024-wide rows."""
    n = u.size
    cols = 1024
    rows = max(1, (n + cols - 1) // cols)
    U = np.full(rows * cols, -1.0, np.float32)
    V = np.full(rows * cols, -1.0, np.float32)
    U[:n], V[:n] = u, v
    out = cv2.remap(img, U.reshape(rows, cols), V.reshape(rows, cols), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.ravel()[:n]


class PlaneParallax:
    def __init__(self, K_cur, K_ref, T_map_cur, T_map_ref, cfg):
        T_rc = np.linalg.inv(np.asarray(T_map_ref)) @ np.asarray(T_map_cur)
        self.R, self.t = T_rc[:3, :3], T_rc[:3, 3]
        self.Kr = np.asarray(K_ref, np.float64)
        self.Kci = np.linalg.inv(np.asarray(K_cur, np.float64))
        self.cfg = cfg
        self.p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / max(cfg.pp_plane_depth_m, 1e-3)])
        self.sigma = np.r_[[np.radians(cfg.pp_rot_sigma_deg)] * 3,
                           cfg.pp_tilt_sigma, cfg.pp_tilt_sigma, cfg.pp_inv_depth_sigma]

    def warp(self, p):
        """3x3 warp, current px -> reference px, for parameters p = [w, m]."""
        H = self.Kr @ rodrigues(p[:3]) @ (self.R + np.outer(self.t, p[3:])) @ self.Kci
        return H / H[2, 2]

    def solve(self, cur_p, ref_p, cur_valid, ref_valid, trace=None):
        """cur_p / ref_p: preprocessed feature images. Returns (W, p)."""
        cfg = self.cfg
        h, w = cur_p.shape[:2]
        step = max(1, int(cfg.pp_step))
        p = self.p0.copy()
        k3 = np.ones((3, 3), np.uint8)
        n_levels = max(1, int(cfg.pyramid_levels))
        finest = min(max(0, int(cfg.pp_finest_level)), n_levels - 1)
        for level in reversed(range(finest, n_levels)):
            s = 0.5 ** level
            size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
            if level:
                cl = cv2.resize(cur_p, size, interpolation=cv2.INTER_AREA)
                rl = cv2.resize(ref_p, size, interpolation=cv2.INTER_AREA)
                cv_l = cv2.resize(cur_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
                rv_l = cv2.resize(ref_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            else:
                cl, rl = cur_p, ref_p
                cv_l, rv_l = cur_valid.astype(np.uint8), ref_valid.astype(np.uint8)
            cv_l = cv2.erode(cv_l, k3)
            rv_l = cv2.erode(rv_l, k3).astype(np.float32)
            gx = cv2.Sobel(rl, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(rl, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            S = _scale_mat(s)
            Si = np.linalg.inv(S)
            ys, xs = np.mgrid[0:size[1]:step, 0:size[0]:step]
            sel = cv_l[ys, xs] > 0
            X = np.stack([xs[sel], ys[sel], np.ones(int(sel.sum()))]).astype(np.float64)
            c_vals = cl[ys[sel], xs[sel]].astype(np.float64)
            a, b = 1.0, 0.0
            n_it = 0
            for n_it in range(1, int(cfg.pp_max_iterations) + 1):
                Y = (S @ self.warp(p) @ Si) @ X
                u, v = Y[0] / Y[2], Y[1] / Y[2]
                ok = ((u >= 0) & (u <= size[0] - 1) & (v >= 0) & (v <= size[1] - 1)
                      & (sample(rv_l, u, v, cv2.INTER_NEAREST) > 0.5))
                if ok.sum() < 50:
                    break
                u, v, Xo = u[ok], v[ok], X[:, ok]
                r_vals = sample(rl, u, v).astype(np.float64)
                gxu, gyu = sample(gx, u, v), sample(gy, u, v)
                J = np.empty((u.size, 8))
                for k in range(6):
                    dp = np.zeros(6)
                    dp[k] = 1e-4
                    Y2 = (S @ self.warp(p + dp) @ Si) @ Xo
                    du = (Y2[0] / Y2[2] - u) / 1e-4
                    dv = (Y2[1] / Y2[2] - v) / 1e-4
                    J[:, k] = a * (gxu * du + gyu * dv)
                J[:, 6] = r_vals
                J[:, 7] = 1.0
                res = c_vals[ok] - (a * r_vals + b)
                scale = 1.4826 * np.median(np.abs(res)) + 1e-6
                uu = np.abs(res) / (HUBER_K * scale)
                wt = np.where(uu <= 1.0, 1.0, 1.0 / np.maximum(uu, 1e-12))
                JW = J * wt[:, None]
                A = J.T @ JW
                g = JW.T @ res
                # MAP: Gaussian priors on the 6 geometric parameters, expressed in
                # units of the residual variance.
                P = scale ** 2 / self.sigma ** 2
                A[np.arange(6), np.arange(6)] += P
                g[:6] -= P * (p - self.p0)
                try:
                    delta = np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    break
                p = p + delta[:6]
                a, b = a + delta[6], b + delta[7]
                if np.max(np.abs(delta[:3])) < 1e-6 and np.max(np.abs(delta[3:6])) < 1e-5:
                    break
            if trace is not None:
                trace.append({'level': level, 'scale': s, 'W': self.warp(p), 'error': '',
                              'iterations': n_it, 'params': p.copy()})
        return self.warp(p), p
