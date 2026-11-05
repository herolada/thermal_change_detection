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
    for features, tol in (('intensity', 1.0), ('edges', 1.5)):  # edges: blurred, less exact
        res = align(cur, valid, ref, valid, H_prior, AlignConfig(features=features))
        assert res.ok, res.reason
        assert not res.fallback
        assert corner_err(res.W, W_true) < tol, features
        assert res.correlation > 0.95


def test_edges_ignore_bright_object(canvas):
    """A large hot object (person) in the reference only must not drag the
    alignment, and must not make the gate reject a correct alignment."""
    valid = np.ones((H, W), np.uint8)
    T_ref, T_cur = pose(rot(0)), pose(rot(4, 1))
    ref = render_view(canvas, T_ref, noise=3, seed=1)
    ref[150:420, 380:470] += 3000.0          # person: ~10x the scene contrast
    cur = render_view(canvas, T_cur, noise=3, seed=2)
    W_true = np.linalg.inv(rotation_homography(K, K, T_cur, T_ref))
    H_prior = rotation_homography(K, K, pose(rot(5.2, 0.5)), T_ref)
    res = align(cur, valid, ref, valid, H_prior, AlignConfig(features='edges'))
    assert res.ok, res.reason
    assert corner_err(res.W, W_true) < 2.0


def test_implausible_refinement_falls_back_to_prior(canvas):
    valid = np.ones((H, W), np.uint8)
    T_ref, T_cur = pose(rot(0)), pose(rot(5, 1))
    ref = render_view(canvas, T_ref, noise=3, seed=1)
    cur = render_view(canvas, T_cur, noise=3, seed=2)
    H_prior = rotation_homography(K, K, pose(rot(6.5, 0.2, 0.8)), T_ref)  # ~14 px off
    cfg = AlignConfig(disp_base_px=5.0)
    res = align(cur, valid, ref, valid, H_prior, cfg, T_map_cur=T_cur, T_map_ref=T_ref,
                K_cur=K, K_ref=K)
    assert res.fallback and res.correction_px > res.bound_px == pytest.approx(5.0)
    assert np.allclose(res.W, res.W0)
    # translation widens the bound: 0.5 m at >= 5 m depth -> +52 px
    T_cur_t = pose(rot(5, 1), (0.5, 0, 0))
    res = align(cur, valid, ref, valid, H_prior, cfg, T_map_cur=T_cur_t, T_map_ref=T_ref,
                K_cur=K, K_ref=K)
    assert not res.fallback and res.bound_px == pytest.approx(5.0 + 520.0 * 0.5 / 5.0)


def test_plane_parallax_recovers_translation_and_heading_error(canvas):
    """Planar scene at 10 m, camera translated by 0.6 m: the rotation-only prior
    is off by the parallax plus a 1 deg heading error; the plane+parallax model
    explains both with its 6 physical parameters."""
    Z = 10.0
    T_ref = pose(rot(0))
    T_cur = pose(rot(3, 0.5), (0.5, 0.05, 0.3))
    # exact plane-induced warp current px -> reference px (plane z = Z in ref frame)
    R, t = T_cur[:3, :3], T_cur[:3, 3]
    n_c = R.T @ np.array([0, 0, 1.0])
    W_true = K @ (R + np.outer(t, n_c) / (Z - t[2])) @ np.linalg.inv(K)
    W_true /= W_true[2, 2]
    M_ref = K_CANVAS @ np.linalg.inv(K)       # ref view px -> canvas px
    ref = render_view(canvas, T_ref, noise=3, seed=1)
    cur = cv2.warpPerspective(canvas, M_ref @ W_true, (W, H),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    cur = cur + np.random.default_rng(2).normal(0, 3, cur.shape).astype(np.float32)
    valid = np.ones((H, W), np.uint8)
    T_meas = pose(rot(4, 0.5), (0.5, 0.05, 0.3))   # 1 deg heading error
    H_prior = rotation_homography(K, K, T_meas, T_ref)
    assert corner_err(np.linalg.inv(H_prior), W_true) > 15
    cfg = AlignConfig(method='plane_parallax', features='edges')
    trace = []
    res = align(cur, valid, ref, valid, H_prior, cfg, trace=trace, T_map_cur=T_meas,
                T_map_ref=T_ref, K_cur=K, K_ref=K)
    assert res.ok and not res.fallback, res.reason
    assert corner_err(res.W, W_true) < 1.0
    assert len(trace) == cfg.pyramid_levels and res.params is not None
    # coarse only (1/4 res): one level solved, a few px at most
    cfg.pp_finest_level = 2
    trace = []
    res = align(cur, valid, ref, valid, H_prior, cfg, trace=trace, T_map_cur=T_meas,
                T_map_ref=T_ref, K_cur=K, K_ref=K)
    assert res.ok and len(trace) == 1 and trace[0]['scale'] == 0.25
    assert corner_err(res.W, W_true) < 3.0


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



def test_change_detect_colder_switch(canvas):
    T = pose(rot(0))
    ref = render_view(canvas, T, noise=3, seed=1)
    cur = render_view(canvas, T, noise=3, seed=2)
    cur = add_blob(cur, 400, 300, 150.0)
    cur = add_blob(cur, 200, 150, -150.0)
    valid = np.ones((H, W), bool)
    kinds = sorted(b.kind for b in compute_change(cur, ref, valid, ChangeConfig()).blobs)
    assert kinds == ['colder', 'warmer']
    trace = {}
    ch = compute_change(cur, ref, valid, ChangeConfig(detect_colder=False), trace=trace)
    assert [b.kind for b in ch.blobs] == ['warmer']
    assert not trace['colder_open'].any()


def test_undistorter_distort_back_roundtrip():
    from thermal_change_detection.core.camera import Undistorter
    Kc = np.array([[409.4, 0, 332.4], [0, 412.6, 262.0], [0, 0, 1]])
    D = np.array([-0.3135, 0.0662, -0.0016, -0.0034])     # strong barrel (eval bag)
    und = Undistorter(Kc, D, (W, H))
    rng = np.random.default_rng(0)
    raw = cv2.GaussianBlur(rng.uniform(0, 1000, (H, W)).astype(np.float32), (0, 0), 3)
    back = und.distort_back(und.undistort(raw))
    inner = np.zeros((H, W), bool)
    inner[60:-60, 80:-80] = True
    assert np.median(np.abs(back - raw)[inner]) < 2.0
    # point mapping is the inverse of the remap
    p = np.array([[50.0, 40.0], [600.0, 480.0], [332.4, 262.0]])
    mx, my = und.distort_maps()
    q = und.distort_points(np.c_[[mx[int(y), int(x)] for x, y in p],
                                 [my[int(y), int(x)] for x, y in p]])
    assert np.abs(q - np.floor(p)).max() < 0.05
    x, y, w, h = und.distort_bbox((0, 0, W, H))
    assert x > 0 and y > 0 and x + w < W and y + h < H   # barrel: full undistorted
                                                         # frame covers less of the raw


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
    heat = np.zeros((H // 2, W // 2, 3), np.uint8)   # false-colour image at half resolution
    heat[..., 0] = 80
    r = p.process(np.clip(img, 0, 65535).astype(np.uint16), 100.0, T_meas, K, D,
                  heatmap=heat)
    assert r.match is not None and r.inserted is None
    assert r.alignment.ok, r.status
    assert r.change is not None and r.reference_view is not None
    kinds = sorted(b.kind for b in r.detections)
    assert kinds.count('warmer') == 2 and kinds.count('hotspot') == 1, r.status
    assert any(abs(b.centroid[0] - 250) < 6 and abs(b.centroid[1] - 200) < 6
               for b in r.detections if b.kind == 'warmer')
    # overlay drawn on the heatmap: blob pixels tinted white, the rest untouched
    assert r.overlay.shape == (H, W, 3)
    assert r.overlay[200, 250, 1] > 80 and r.overlay[200, 250, 0] > 80
    assert tuple(r.overlay[100, 320]) == (80, 0, 0)
    p.close()


def test_params_yaml_in_sync_with_config():
    import os
    import yaml
    from thermal_change_detection.core.config import _flatten_dict
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'params.yaml')
    with open(path) as f:
        params = _flatten_dict(yaml.safe_load(f)['thermal_change_detection']['ros__parameters'])
    node_only = {'map_frame', 'camera_frame', 'frame_is_optical', 'tf_timeout',
                 'heatmap_topic', 'heatmap_max_dt'}
    defaults = flatten(PipelineConfig())
    assert set(params) - node_only == set(defaults)
    for k, v in defaults.items():
        assert params[k] == v, k


# --------------------------------------------------------------------------- stability

def _blob_map(rects, shape=(120, 160)):
    """rects: list of (kind, x, y, w, h) -> (blobs, labels)."""
    from thermal_change_detection.core.change import Blob
    labels = np.zeros(shape, np.int32)
    blobs = []
    for kind, x, y, w, h in rects:
        labels[y:y + h, x:x + w] = len(blobs) + 1
        blobs.append(Blob(kind, (x, y, w, h), (x + w / 2, y + h / 2), w * h, 0.0, 0.0, 0.0))
    return blobs, labels


def test_stability_measure_keeps_steady_and_drops_changing():
    from thermal_change_detection.core.config import StabilityConfig
    from thermal_change_detection.core.stability import measure
    cfg = StabilityConfig(max_size_change=0.5, match_radius_px=3, min_coverage=0.7)
    cur, cur_l = _blob_map([('warmer', 10, 10, 20, 20),    # steady
                            ('warmer', 60, 10, 20, 20),    # shrank to a quarter
                            ('warmer', 110, 10, 10, 10),   # vanished
                            ('colder', 10, 60, 20, 20),    # split in two in aux
                            ('warmer', 60, 60, 20, 20)])   # aux has a COLDER blob there
    aux, aux_l = _blob_map([('warmer', 11, 11, 20, 19),
                            ('warmer', 62, 12, 10, 10),
                            ('colder', 10, 60, 9, 20), ('colder', 21, 60, 9, 20),
                            ('colder', 60, 60, 20, 20)])
    valid = np.ones_like(cur_l, bool)
    measure(cur, cur_l, aux, aux_l, valid, cfg)
    assert [b.stable for b in cur] == [True, False, False, True, False]
    assert cur[0].size_change < 0.1 and cur[2].size_change == 1.0


def test_stability_not_judged_outside_aux_overlap():
    from thermal_change_detection.core.config import StabilityConfig
    from thermal_change_detection.core.stability import drop_unstable, measure
    from thermal_change_detection.core.change import ChangeResult
    cfg = StabilityConfig(min_coverage=0.7)
    cur, cur_l = _blob_map([('warmer', 10, 10, 20, 20), ('warmer', 100, 10, 20, 20)])
    aux, aux_l = _blob_map([])
    valid = np.ones_like(cur_l, bool)
    valid[:, :90] = False                  # first blob outside the aux overlap
    measure(cur, cur_l, aux, aux_l, valid, cfg)
    assert cur[0].stable and not cur[1].stable
    ch = ChangeResult(None, None, valid, None, 1.0, 1.0, 0.0, cur, cur_l)
    removed = drop_unstable(ch)
    assert len(removed) == 1 and len(ch.blobs) == 1
    assert set(np.unique(ch.labels)) == {0, 1} and ch.labels[15, 15] == 1


def test_frame_buffer_pick():
    from thermal_change_detection.core.stability import FrameBuffer
    fb = FrameBuffer(1.0)
    for i in range(40):
        fb.push(i / 30.0, i)
    s, item = fb.pick(39 / 30.0, 0.2)
    assert item == 33
    assert fb.pick(39 / 30.0, 5.0) is None


# --------------------------------------------------------------------------- frame selection

def _drive(sel, xs, dt=1 / 30.0, t0=100.0, y=0.0):
    """Push frames along the x axis (camera looking along z); returns the selections."""
    out = []
    for i, x in enumerate(xs):
        r = sel.push(t0 + i * dt, pose(rot(0), (x, y, 0)), i)
        if r is not None:
            out.append(r)
    return out


def _store_with_keyframes(tmp_path, xs):
    s = KeyframeStore(StoreConfig(storage_dir=str(tmp_path), min_keyframe_age_s=10,
                                  r_match=1.0))
    for x in xs:
        s.insert(np.zeros((4, 4), np.uint16), pose(rot(0), (x, 0, 0)), K, 0.0)
    return s


def test_frame_selector_picks_frame_closest_to_each_keyframe(tmp_path):
    from thermal_change_detection.core.config import SelectConfig
    from thermal_change_detection.core.frame_select import FrameSelector
    kf_x = [0.37, 1.37, 2.37]
    store = _store_with_keyframes(tmp_path, kf_x)
    sel = FrameSelector(SelectConfig(mode='closest', rise_margin_m=0.05), store)
    xs = np.arange(-0.5, 3.5, 0.03)                     # 0.9 m/s at 30 Hz
    got = _drive(sel, xs, y=0.2)                        # driving 0.2 m beside the old path
    picked = [(s.kf_id, xs[s.item]) for s in got if s.kf_id is not None]
    assert [k for k, _ in picked] == [0, 1, 2]
    for k, x in picked:
        assert abs(x - kf_x[k]) <= 0.015 + 1e-9         # closest frame (half a frame step)
    assert all(s.reason == 'passed' for s in got if s.kf_id is not None)
    assert all(s.reason == 'no_ref' for s in got if s.kf_id is None)   # past the route end
    store.close()


def test_frame_selector_falls_back_to_periodic_without_reference(tmp_path):
    from thermal_change_detection.core.config import SelectConfig
    from thermal_change_detection.core.frame_select import FrameSelector
    store = _store_with_keyframes(tmp_path, [])
    sel = FrameSelector(SelectConfig(mode='closest', period=1.0), store)
    got = _drive(sel, np.linspace(0, 3, 91))            # 3 s
    assert [s.reason for s in got] == ['no_ref'] * 4
    assert np.allclose(np.diff([s.stamp for s in got]), 1.0, atol=1 / 30.0)
    store.close()


def test_frame_selector_standing_still_uses_max_period(tmp_path):
    from thermal_change_detection.core.config import SelectConfig
    from thermal_change_detection.core.frame_select import FrameSelector
    store = _store_with_keyframes(tmp_path, [0.0])
    sel = FrameSelector(SelectConfig(mode='closest', max_period=2.0), store)
    got = _drive(sel, np.zeros(30 * 7))                 # 7 s at the keyframe
    assert len(got) == 4 and all(s.reason == 'timeout' for s in got)
    assert np.all(np.abs(np.diff([s.stamp for s in got]) - 2.0) <= 1.5 / 30.0)
    store.close()


def test_frame_selector_periodic_mode(tmp_path):
    from thermal_change_detection.core.config import SelectConfig
    from thermal_change_detection.core.frame_select import FrameSelector
    store = _store_with_keyframes(tmp_path, [0.0])
    sel = FrameSelector(SelectConfig(mode='periodic', period=0.5), store)
    got = _drive(sel, np.zeros(60))
    assert [s.item for s in got] == [0, 15, 30, 45]
    store.close()
