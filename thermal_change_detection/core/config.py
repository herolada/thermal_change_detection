"""Configuration dataclasses for the change detection pipeline.

Every field is exposed 1:1 as a ROS parameter named ``<section>.<field>``
(e.g. ``store.r_match``), so ``config/params.yaml`` mirrors this file.
"""
from dataclasses import dataclass, field, fields, is_dataclass

import yaml


@dataclass
class StoreConfig:
    # A new keyframe is stored when no keyframe lies within BOTH of these.
    r_insert: float = 0.5               # [m]
    theta_insert_deg: float = 5.0       # [deg] angle between optical axes
    # A keyframe is used as reference when it lies within BOTH of these ...
    r_match: float = 1.5                # [m]
    theta_match_deg: float = 15.0       # [deg]
    # ... and is at least this old (prevents comparing against the frame
    # stored a second ago on the same lap).
    min_keyframe_age_s: float = 60.0
    # Ranking of matching keyframes (and of frames by core/frame_select.py):
    # score = distance [m] + angle_weight_m_per_deg * angle [deg]. Translation
    # dominates: rotation is compensated by the homography, parallax is not.
    angle_weight_m_per_deg: float = 0.01
    # Keyframe images go to <storage_dir>/run_<timestamp>_<pid>/.
    storage_dir: str = '/tmp/thermal_change_detection'
    cleanup_on_exit: bool = True        # delete this run's directory on shutdown


@dataclass
class AlignConfig:
    # Refinement of the pose prior (rotation homography):
    #   'ecc'            OpenCV ECC, free homography
    #   'plane_parallax' 6-parameter physical model with the known translation
    #                    (scene plane + small rotation correction), robust
    #   'none'           pose prior only
    method: str = 'ecc'
    # Images the refinement and the correlation gate work on:
    #   'intensity' (raw), 'gradient' (gradient magnitude),
    #   'edges' (saturated gradient magnitude: hot objects cannot dominate)
    features: str = 'edges'
    pyramid_levels: int = 3             # 3 -> solve at 1/4, 1/2, 1/1 resolution
    # ECC
    max_iterations: int = 100           # per pyramid level
    eps: float = 1e-5                   # convergence threshold
    gauss_filt_size: int = 5            # internal blur, odd, 1 = off
    # plane_parallax
    pp_max_iterations: int = 30         # Gauss-Newton iterations per pyramid level
    pp_step: int = 2                    # use every n-th pixel (per axis)
    pp_finest_level: int = 0            # stop at this pyramid level: 0 = full res,
                                        # 1 = 1/2, 2 = 1/4 (much faster, less exact)
    pp_rot_sigma_deg: float = 1.0       # prior std of the rotation correction
    pp_plane_depth_m: float = 20.0      # prior: scene plane facing the camera at this depth
    pp_inv_depth_sigma: float = 0.1     # [1/m] prior std of the plane's inverse depth
    pp_tilt_sigma: float = 0.03         # [1/m] prior std of the plane's tilt components
    # Physical bound on the refinement: if it moves any image point further from
    # the pose prior than disp_base_px + f * |t| / disp_min_depth_m (rotation
    # error allowance + parallax of the nearest expected scene point at
    # camera translation t), the refinement is discarded and the prior is used.
    disp_base_px: float = 40.0
    disp_min_depth_m: float = 5.0
    # Quality gate: the frame is rejected (no change map) if any check fails.
    min_correlation: float = 0.5        # correlation of the aligned feature images
    min_overlap: float = 0.3            # overlapping fraction of the current image


@dataclass
class ChangeConfig:
    border_erode_px: int = 3            # shrink the overlap mask (interpolation borders)
    gain_fit_samples: int = 20000       # pixels sampled for the robust gain/offset fit
    huber_k: float = 1.5                # Huber threshold in robust-sigma units
    nbhd_radius: int = 2                # misalignment tolerance [px], 0 = plain difference
    blur_sigma: float = 2.0             # Gaussian smoothing of the difference [px]
    min_sigma: float = 2.0              # floor of the noise estimate [image units]
    z_low: float = 3.0                  # hysteresis thresholds in noise-sigma units
    z_high: float = 6.0
    open_kernel: int = 5                # morphological opening kernel [px], 0 = off
    min_blob_area: int = 30             # [px]
    detect_colder: bool = True          # False: only 'warmer' blobs (colder pixels are
                                        #   also hidden in the overlay heatmap)


@dataclass
class HotspotConfig:
    # Absolute threshold on the (undistorted) image values, e.g. Boson RAW16
    # counts. Needs to be calibrated for the camera; 0 disables the detector.
    abs_threshold: float = 0.0
    open_kernel: int = 3
    min_blob_area: int = 10


@dataclass
class PersistenceConfig:
    # A blob is "confirmed" once it was seen in this many consecutive processed
    # frames. 1 disables the check.
    required_frames: int = 2
    max_gap_s: float = 5.0              # longer gap between frames resets the counts
    radius_px: int = 25                 # association tolerance after warping


@dataclass
class StabilityConfig:
    # Short-baseline check (core/stability.py): each blob is compared with the
    # blobs of a frame from lookback_s earlier, compared with the same
    # keyframe. Blobs whose size changes too much (parallax, occlusion,
    # moving branches) are dropped. Costs one extra alignment per frame with
    # blobs.
    enabled: bool = False
    lookback_s: float = 0.2             # [s] age of the auxiliary frame
    max_size_change: float = 0.5        # 1 - min(A)/max(A) above this -> unstable
    match_radius_px: int = 10           # blobs overlapping within this radius are associated
    min_coverage: float = 0.7           # blobs less covered by the aux frame are kept


@dataclass
class SelectConfig:
    # Which incoming frames are processed (core/frame_select.py):
    #   'periodic' the first frame after `period` (ignores the keyframes)
    #   'closest'  per keyframe, the frame closest to it (lowest store score),
    #              decided once the score has risen again (we drove past it);
    #              without a matching keyframe (first lap, off the path) it
    #              falls back to 'periodic' so keyframes still get stored.
    mode: str = 'closest'
    period: float = 1.0                 # [s] 'periodic' / no-reference spacing
    min_period: float = 0.2             # [s] never process frames closer than this
    max_period: float = 3.0             # [s] process the best frame so far after this
                                        #     (robot standing still near a keyframe)
    rise_margin_m: float = 0.05         # [score units ~ m] rise above the minimum
                                        #     that marks the keyframe as passed


@dataclass
class VizConfig:
    gray_low_pct: float = 1.0           # percentiles for grayscale background
    gray_high_pct: float = 99.0
    z_min: float = 3.0                  # |z| below this is fully transparent
    z_max: float = 12.0                 # |z| at which colour/alpha saturate
    alpha_max: float = 0.75
    draw_unconfirmed: bool = True       # thin boxes for not-yet-confirmed blobs
    # True: ~/overlay is drawn on the original (distorted) camera image: the
    # change map and blob boxes are mapped back through the lens model.
    # False: it is drawn on the undistorted (pinhole) image the pipeline uses.
    # Only the overlay changes; ~/diff, ~/reference_warped and ~/detections
    # stay in undistorted coordinates.
    distort_overlay: bool = False
    outline_thickness: int = 1          # [px] blob outline
    # Status-line "quality" = noise sigma mapped linearly to 100 % (at or below
    # quality_sigma_good) .. 0 % (at or above quality_sigma_bad) [image units].
    quality_sigma_good: float = 8.0
    quality_sigma_bad: float = 40.0


@dataclass
class PipelineConfig:
    store: StoreConfig = field(default_factory=StoreConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    hotspot: HotspotConfig = field(default_factory=HotspotConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    viz: VizConfig = field(default_factory=VizConfig)


def flatten(cfg, prefix=''):
    """Nested dataclass -> {'section.field': value}."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        key = prefix + f.name
        if is_dataclass(v):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def update_from_flat(cfg, flat):
    """Set fields from {'section.field': value}, casting to the field's type."""
    for key, value in flat.items():
        obj = cfg
        *path, name = key.split('.')
        for p in path:
            obj = getattr(obj, p)
        if not hasattr(obj, name):
            raise KeyError(f'unknown parameter {key}')
        current = getattr(obj, name)
        setattr(obj, name, type(current)(value))
    return cfg


def load_ros_params_yaml(path, cfg=None):
    """Read a ROS 2 params file (``<node>: ros__parameters: {...}``) into a config."""
    cfg = cfg or PipelineConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    known = flatten(cfg)
    for node_params in data.values():
        params = node_params.get('ros__parameters', {})
        flat = _flatten_dict(params)
        update_from_flat(cfg, {k: v for k, v in flat.items() if k in known})
    return cfg


def _flatten_dict(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten_dict(v, f'{prefix}{k}.'))
        else:
            out[prefix + k] = v
    return out
