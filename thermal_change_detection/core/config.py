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
    # Keyframe images go to <storage_dir>/run_<timestamp>_<pid>/.
    storage_dir: str = '/tmp/thermal_change_detection'
    cleanup_on_exit: bool = True        # delete this run's directory on shutdown


@dataclass
class AlignConfig:
    ecc_enabled: bool = True            # False -> pose-only (rotation homography)
    pyramid_levels: int = 3             # 3 -> solve at 1/4, 1/2, 1/1 resolution
    max_iterations: int = 100           # per pyramid level
    eps: float = 1e-5                   # ECC convergence threshold
    gauss_filt_size: int = 5            # ECC internal blur, odd, 1 = off
    use_gradient: bool = False          # align gradient magnitude instead of intensities
    # Quality gate: the frame is rejected (no change map) if any check fails.
    min_correlation: float = 0.6        # final ECC correlation coefficient
    max_correction_px: float = 80.0     # max shift of image corners vs. the pose prior
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
class VizConfig:
    gray_low_pct: float = 1.0           # percentiles for grayscale background
    gray_high_pct: float = 99.0
    z_min: float = 3.0                  # |z| below this is fully transparent
    z_max: float = 12.0                 # |z| at which colour/alpha saturate
    alpha_max: float = 0.75
    draw_unconfirmed: bool = True       # thin boxes for not-yet-confirmed blobs


@dataclass
class PipelineConfig:
    store: StoreConfig = field(default_factory=StoreConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    hotspot: HotspotConfig = field(default_factory=HotspotConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
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
