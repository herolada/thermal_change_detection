"""Synthetic tests of the ROS-free core.

The scene is a large canvas "at infinity" seen by a pinhole camera that only
rotates, so the rotation homography is the exact ground-truth warp.
"""
import cv2
import numpy as np
import pytest

from thermal_change_detection.core.align import align, warp_to_current
from thermal_change_detection.core.change import compute_change
from thermal_change_detection.core.config import (AlignConfig, ChangeConfig, PipelineConfig,
                                                  PersistenceConfig, StoreConfig, flatten,
                                                  update_from_flat)
from thermal_change_detection.core.geometry import rotation_homography
from thermal_change_detection.core.keyframe_store import KeyframeStore
from thermal_change_detection.core.persistence import PersistenceTracker
from thermal_change_detection.core.pipeline import ChangeDetectionPipeline

W, H = 640, 512
K = np.array([[520.0, 0, W / 2 - 0.5], [0, 520.0, H / 2 - 0.5], [0, 0, 1]])
CW, CH = 1600, 1200
K_CANVAS = np.array([[520.0, 0, CW / 2], [0, 520.0, CH / 2], [0, 0, 1]])


def make_canvas(seed=1):
    rng = np.random.default_rng(seed)
    c = np.full((CH, CW), 22000.0, np.float32)
    low = cv2.resize(rng.normal(0, 1, (12, 16)).astype(np.float32), (CW, CH),
                     interpolation=cv2.INTER_CUBIC)
    c += 300 * low
    for _ in range(60):  # warm/cold rectangles = edges for alignment
        x, y = rng.integers(0, CW - 120), rng.integers(0, CH - 120)
        w, h = rng.integers(20, 120, size=2)
        c[y:y + h, x:x + w] += rng.uniform(-400, 400)
    c = cv2.GaussianBlur(c, (0, 0), 1.5)
    return c


def rot(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Rotation of an optical-frame camera (yaw about y, pitch about x, roll about z)."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def pose(R, t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def render_view(canvas, T, noise=0.0, seed=0):
    M = K_CANVAS @ T[:3, :3] @ np.linalg.inv(K)  # view px -> canvas px
    img = cv2.warpPerspective(canvas, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape).astype(np.float32)
    return img


def add_blob(img, cx, cy, amp, sigma=6.0):
    yy, xx = np.mgrid[0:H, 0:W]
    return img + amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(
        np.float32)


def corner_err(Wa, Wb):
    c = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], float).T
    a, b = Wa @ c, Wb @ c
    return np.max(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2], axis=0))


@pytest.fixture(scope='module')
def canvas():
    return make_canvas()


def test_config_roundtrip():
    cfg = PipelineConfig()
    flat = flatten(cfg)
    assert flat['store.r_match'] == 1.5
    update_from_flat(cfg, {'store.r_match': 2, 'align.pyramid_levels': 2.0})
    assert cfg.store.r_match == 2.0 and isinstance(cfg.store.r_match, float)
    assert cfg.align.pyramid_levels == 2 and isinstance(cfg.align.pyramid_levels, int)


def test_rotation_homography_matches_rendering(canvas):
    T_ref, T_cur = pose(rot(0)), pose(rot(4, 1, 0.5))
    ref, cur = render_view(canvas, T_ref), render_view(canvas, T_cur)
    Hcr = rotation_homography(K, K, T_cur, T_ref)
    ref_w = warp_to_current(ref, np.linalg.inv(Hcr), (W, H))
    inner = (slice(60, H - 60), slice(80, W - 80))
    assert np.abs(ref_w[inner] - cur[inner]).mean() < 5.0


def test_ecc_corrects_pose_error(canvas):
    valid = np.ones((H, W), np.uint8)
    T_ref, T_cur = pose(rot(0)), pose(rot(5, 1))
    ref = render_view(canvas, T_ref, noise=3, seed=1)
    cur = render_view(canvas, T_cur, noise=3, seed=2) * 1.04 + 250  # camera drift
    W_true = np.linalg.inv(rotation_homography(K, K, T_cur, T_ref))
    T_cur_noisy = pose(rot(6.5, 0.2, 0.8))  # ~1.5 deg heading error -> ~14 px
    H_prior = rotation_homography(K, K, T_cur_noisy, T_ref)
    assert corner_err(np.linalg.inv(H_prior), W_true) > 10
    res = align(cur, valid, ref, valid, H_prior, AlignConfig())
    assert res.ok, res.reason
    assert corner_err(res.W, W_true) < 1.0
    assert res.correlation > 0.95


def test_change_detects_hot_blob_and_ignores_drift(canvas):
    T = pose(rot(0))
    ref = render_view(canvas, T, noise=3, seed=1)
    cur = render_view(canvas, T, noise=3, seed=2)
    cur = add_blob(cur, 400, 300, 150.0) * 1.05 + 300.0
    # 1 px misalignment of the reference: should not produce edge artefacts
    ref_shift = cv2.warpAffine(ref, np.float32([[1, 0, 1], [0, 1, 0]]), (W, H))
    valid = np.ones((H, W), bool)
    valid[:, :2] = False
    ch = compute_change(cur, ref_shift, valid, ChangeConfig())
    assert abs(ch.gain - 1.05) < 0.02
    assert len(ch.blobs) == 1, [(b.kind, b.bbox) for b in ch.blobs]
    b = ch.blobs[0]
    assert b.kind == 'warmer'
    assert abs(b.centroid[0] - 400) < 5 and abs(b.centroid[1] - 300) < 5
    assert np.isnan(ch.diff[:, 0]).all()


def test_store_insert_and_query(tmp_path):
    s = KeyframeStore(StoreConfig(storage_dir=str(tmp_path), min_keyframe_age_s=10))
    img = np.zeros((H, W), np.uint16)
    T0 = pose(rot(0))
    assert s.needs_insert(T0)
    s.insert(img, T0, K, 0.0)
    assert not s.needs_insert(pose(rot(2), (0.3, 0, 0)))
    assert s.needs_insert(pose(rot(0), (0.8, 0, 0)))       # too far
    assert s.needs_insert(pose(rot(10), (0, 0, 0)))         # turned too much
    assert s.query(pose(rot(0), (1.0, 0, 0)), 5.0) is None  # too young
    m = s.query(pose(rot(3), (1.0, 0, 0)), 20.0)
    assert m is not None and abs(m.distance - 1.0) < 1e-9
    assert s.query(pose(rot(20)), 20.0) is None             # wrong direction
    assert s.load(m.keyframe).dtype == np.uint16
    s.close()
    assert not (tmp_path / s.run_dir).exists()


def test_persistence_tracker():
    from thermal_change_detection.core.change import Blob
    tr = PersistenceTracker(PersistenceConfig(required_frames=2, radius_px=10))
    labels = np.zeros((H, W), np.int32)
    labels[100:120, 100:120] = 1
    T0, T1 = pose(rot(0)), pose(rot(1))
    b0 = [Blob('warmer', (100, 100, 20, 20), (110, 110), 400, 1, 1, 1)]
    tr.update(b0, labels, 0.0, T0, K)
    assert not b0[0].confirmed
    # same world point, camera rotated 1 deg -> moves ~9 px in the image
    labels1 = np.zeros((H, W), np.int32)
    labels1[100:120, 91:111] = 1
    b1 = [Blob('warmer', (91, 100, 20, 20), (101, 110), 400, 1, 1, 1)]
    tr.update(b1, labels1, 1.0, T1, K)
    assert b1[0].confirmed and b1[0].persistence == 2
    # gap too long -> reset
    b2 = [Blob('warmer', (91, 100, 20, 20), (101, 110), 400, 1, 1, 1)]
    tr.update(b2, labels1, 100.0, T1, K)
    assert b2[0].persistence == 1


def test_pipeline_end_to_end(canvas, tmp_path):
    cfg = PipelineConfig()
    cfg.store.storage_dir = str(tmp_path)
    cfg.store.min_keyframe_age_s = 30.0
    cfg.persistence.required_frames = 1
    p = ChangeDetectionPipeline(cfg)
    D = np.zeros(5)

    # Lap 1: store keyframes.
    T0 = pose(rot(0))
    r = p.process(render_view(canvas, T0, 3, 1).astype(np.uint16), 0.0, T0, K, D)
    assert r.inserted is not None and r.match is None and r.change is None
    assert r.overlay.shape == (H, W, 3)

    # Lap 2: 0.4 m away, rotated by 3 deg, with a 1.2 deg pose error and a new warm object.
    T_true = pose(rot(3, 0.5), (0.4, 0, 0))
    T_meas = pose(rot(4.2, 0.5), (0.4, 0, 0))
    base = render_view(canvas, T_true, 3, 2)
    cfg.hotspot.abs_threshold = float(base.max()) + 500.0
    img = add_blob(base, 250, 200, 200.0)
    img = add_blob(img, 500, 400, 6000.0, sigma=4)  # very hot -> also a hotspot
    r = p.process(np.clip(img, 0, 65535).astype(np.uint16), 100.0, T_meas, K, D)
    assert r.match is not None and r.inserted is None
    assert r.alignment.ok, r.status
    assert r.change is not None and r.reference_view is not None
    kinds = sorted(b.kind for b in r.detections)
    assert kinds.count('warmer') == 2 and kinds.count('hotspot') == 1, r.status
    assert any(abs(b.centroid[0] - 250) < 6 and abs(b.centroid[1] - 200) < 6
               for b in r.detections if b.kind == 'warmer')
    p.close()


def test_params_yaml_in_sync_with_config():
    import os
    import yaml
    from thermal_change_detection.core.config import _flatten_dict
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'params.yaml')
    with open(path) as f:
        params = _flatten_dict(yaml.safe_load(f)['thermal_change_detection']['ros__parameters'])
    node_only = {'map_frame', 'camera_frame', 'frame_is_optical', 'process_period', 'tf_timeout'}
    defaults = flatten(PipelineConfig())
    assert set(params) - node_only == set(defaults)
    for k, v in defaults.items():
        assert params[k] == v, k
