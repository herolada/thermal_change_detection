# thermal_change_detection

Revisit-based change detection for a robot patrolling a repeated route. Every
processed thermal image is compared with a keyframe taken at (nearly) the
same pose on an earlier lap; differences are published as a heatmap overlay
and as detections.

## Pipeline (per processed image, ~1 Hz)

0. **Frame selection** (`select.*`, `core/frame_select.py`): the TF pose of
   *every* incoming image is looked up (~0.2 ms each) and scored against the
   keyframes like in 2. In `mode: closest`, the image processed for a keyframe
   is the one taken closest to it: the decision is made once the score rose
   `rise_margin_m` above its minimum (we drove past), the best match switched
   to another keyframe, or the route was left (`min_period` / `max_period`
   bound the spacing). On the first lap / off the route (no matching keyframe)
   an image is taken every `period`, so keyframes end up ~`v * period` apart
   and the next lap processes ~one image per keyframe, near its pose. Adds
   ~`rise_margin_m / v` latency (plus TF lag). `mode: periodic` = first image
   after `period` (the old behaviour).
1. **Pose**: TF `map <- camera` at the image stamp, lens undistortion from `camera_info`.
2. **Retrieval**: best keyframe within `r_match` / `theta_match_deg` that is at
   least `min_keyframe_age_s` old, ranked by
   `distance + angle_weight_m_per_deg * angle` (translation dominates: rotation
   is compensated by the homography, parallax is not). Independent of that, the image becomes a new
   keyframe if no keyframe lies within `r_insert` / `theta_insert_deg`
   (first lap builds the map, later laps fill gaps). Images are 16-bit PNGs on
   disk, poses in RAM. Every launch starts with an empty map.
3. **Alignment**: rotation homography `K R K^-1` from the poses, refined
   (`align.method`) by pyramidal ECC (free homography) or by the
   plane + parallax model (below), on `align.features` (default `edges`: a
   saturated gradient magnitude, so hot objects such as people cannot dominate
   the alignment or the correlation gate). A refinement that moves any image
   point further from the prior than `disp_base_px + f·|t| / disp_min_depth_m`
   is physically implausible and is discarded (the frame is compared with the
   pose prior alone). Rejected if the overlap or the feature correlation fails
   the gate.
4. **Change**: robust gain/offset fit (camera drift, global warming),
   misalignment-tolerant signed difference, smoothing, z-score w.r.t. robust
   noise sigma, hysteresis + opening + connected components -> `warmer` /
   `colder` blobs (`change.detect_colder: false` keeps only `warmer`).
5. **Hotspots**: reference-free absolute threshold (`hotspot.abs_threshold`).
6. **Persistence**: blobs are confirmed after `persistence.required_frames`
   consecutive frames.

Optional, between 4. and 6.: **stability check** (`stability.enabled`, off by
default), see below.

All logic lives in `thermal_change_detection/core/` (no ROS imports);
`node.py` is a thin wrapper.

### Plane + parallax alignment (`align.method: plane_parallax`)

The two images are taken from known poses, so instead of estimating an
arbitrary homography (8 free parameters) the refinement only estimates what
localisation cannot know:

    W(p) = K_ref · dR(w) · (R + t mᵀ) · K_cur⁻¹      (current px → reference px)

* `R, t`: rotation and translation from the current to the reference camera,
  from the poses. Fixed.
* `m = n / d`: the scene plane `nᵀX = d` (3 parameters). `(R + t mᵀ)` is the
  exact homography induced by that plane, so parallax can only appear along
  the direction the camera really moved, and in proportion to how far it
  moved; with `t ≈ 0`, `m` has no effect.
* `w`: a small rotation correction (3 parameters) for heading, extrinsic
  calibration or image/TF timing errors.

The parameters are solved by robust Gauss-Newton (Huber weights, so pixels
that disagree, such as a person present in only one image, are
down-weighted), coarse to fine, with Gaussian priors that pull `w` towards 0
(`pp_rot_sigma_deg`) and the plane towards one facing the camera at
`pp_plane_depth_m`. The fitted `w` per frame (shown in the debug report)
measures how far the pose prior is off. Cost is about 1 s per frame vs.
0.3 s for ECC.

### Stability check (`stability.*`, experimental, off by default)

Against false positives from parallax, changing occlusion and moving
branches: a frame from `lookback_s` earlier is compared with the *same*
keyframe, its blobs are mapped into the current image through the keyframe,
and blobs whose area changes too much (`1 - min/max > max_size_change`,
vanished = 1) are dropped (thin gray boxes in the overlay, section 5 of the
debug report). Overlapping blobs are grouped so splits/merges are compared as
a whole. Costs one extra alignment per frame with blobs. Details in
`core/stability.py`.

Findings on `g4s_demo_2` (camera ~30 Hz, robot ~0.9 m/s; 576 blobs, 360
confirmed without the check):

| lookback | camera baseline | blobs dropped | confirmed |
|---|---|---|---|
| 0.1 s | 9 cm | 8 % | 331 |
| 0.2 s | 19 cm | 16 % | 306 |
| 0.5 s | 47 cm | 29 % | 257 |

Only 0.5 s removes the tree/sky-edge clutter (0.1-0.2 s is too short a
baseline for the parallax to change). Known issues at 0.5 s: fragments of
walking pedestrian groups are dropped (they move beyond `match_radius_px`
and can flip warmer/colder); dense clusters of fragments are grouped and
their total areas average out (so they survive); no effect when the robot
stands still. Ideas not tried yet: per-blob (non-transitive) matching,
polarity-agnostic matching with a radius growing with the lookback, and
requiring instability at two lookbacks.

## Topics

| | Topic | Type |
|---|---|---|
| in | `image` (default remap `/flir_boson/image_raw`) | `sensor_msgs/Image`, mono16 preferred |
| in | `camera_info` (default `/flir_boson/camera_info`) | `sensor_msgs/CameraInfo` |
| in | param `heatmap_topic` (optional, off if empty) | `sensor_msgs/Image`, false-colour view of the same camera (same aspect ratio, paired by stamp within `heatmap_max_dt`); `~/overlay` is then drawn on it: white = warmer, cyan = colder, green = hotspot (blob outlines) |
| out | `~/overlay` | bgr8: red/yellow warmer, blue/cyan colder, hatched = no overlap; undistorted image, or the original one with `viz.distort_overlay: true` |
| out | `~/diff` | 32FC1 signed difference in image units, NaN outside overlap |
| out | `~/reference_warped` | mono8 aligned reference, same scaling as overlay |
| out | `~/detections` | `vision_msgs/Detection2DArray`, class `warmer`/`colder`/`hotspot` |

Overlay text. Status line, e.g. `kf54 transl=0.30m rot=2.4deg match=87% gain=1.003 quality=59%`:
the matched keyframe, its translation / viewing-angle difference to the current pose,
`match` = correlation of the aligned images, `gain` = fitted `cur ≈ gain·ref + offset`,
`quality` = noise sigma mapped to 100 % (≤ `viz.quality_sigma_good`) .. 0 %
(≥ `viz.quality_sigma_bad`). Blob labels (`+33%`): mean difference of the blob in % of
the scene contrast (the `gray_low_pct`..`gray_high_pct` intensity range of the current image).

Run the Boson in **`video_mode:=RAW16`**: AGC modes rescale every frame, which
breaks the comparison.

## Usage

```bash
ros2 launch thermal_change_detection thermal_change_detection.launch.py \
    image:=/flir_boson/image_raw camera_info:=/flir_boson/camera_info

# Offline tuning on a bag (writes overlay PNGs + log.csv):
ros2 run thermal_change_detection offline_eval <bag> -o /tmp/tcd_out \
    --params src/thermal_change_detection/config/params.yaml
```

Step-by-step HTML debug report on a bag (same pipeline and subsampling,
with every intermediate image): keyframe/frame pairs with the pose change,
the pose-prior homography, ECC per pyramid level, the quality gate (why a
frame is `REJECTED`), and each change-detection step down to the blobs.
Open `index.html` in a browser; `--detail 50:80` limits the per-frame pages.

```bash
ros2 run thermal_change_detection debug_report ~/bags/bag_20261104_130658 -o /tmp/tcd_report \
    --params src/thermal_change_detection/config/params.yaml \
    --image /camera_front_thermo/image_raw/compressed \
    --camera-info /camera_front_thermo/camera_info --map-frame FP_ENU0
```

Tests: `python3 -m pytest src/thermal_change_detection/test`
