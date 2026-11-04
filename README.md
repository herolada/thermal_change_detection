# thermal_change_detection

Revisit-based change detection for a robot patrolling a repeated route. Every
processed thermal image is compared with a keyframe taken at (nearly) the
same pose on an earlier lap; differences are published as a heatmap overlay
and as detections.

## Pipeline (per image, ~1 Hz)

1. **Pose**: TF `map <- camera` at the image stamp, lens undistortion from `camera_info`.
2. **Retrieval**: best keyframe within `r_match` / `theta_match_deg` that is at
   least `min_keyframe_age_s` old. Independent of that, the image becomes a new
   keyframe if no keyframe lies within `r_insert` / `theta_insert_deg`
   (first lap builds the map, later laps fill gaps). Images are 16-bit PNGs on
   disk, poses in RAM. Every launch starts with an empty map.
3. **Alignment**: rotation homography `K R K^-1` from the poses, refined by
   pyramidal ECC (homography). Rejected if correlation / overlap / correction
   from the prior fails the gate.
4. **Change**: robust gain/offset fit (camera drift, global warming),
   misalignment-tolerant signed difference, smoothing, z-score w.r.t. robust
   noise sigma, hysteresis + opening + connected components -> `warmer` /
   `colder` blobs.
5. **Hotspots**: reference-free absolute threshold (`hotspot.abs_threshold`).
6. **Persistence**: blobs are confirmed after `persistence.required_frames`
   consecutive frames.

All logic lives in `thermal_change_detection/core/` (no ROS imports);
`node.py` is a thin wrapper.

## Topics

| | Topic | Type |
|---|---|---|
| in | `image` (default remap `/flir_boson/image_raw`) | `sensor_msgs/Image`, mono16 preferred |
| in | `camera_info` (default `/flir_boson/camera_info`) | `sensor_msgs/CameraInfo` |
| out | `~/overlay` | bgr8: red/yellow warmer, blue/cyan colder, hatched = no overlap |
| out | `~/diff` | 32FC1 signed difference in image units, NaN outside overlap |
| out | `~/reference_warped` | mono8 aligned reference, same scaling as overlay |
| out | `~/detections` | `vision_msgs/Detection2DArray`, class `warmer`/`colder`/`hotspot` |

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
