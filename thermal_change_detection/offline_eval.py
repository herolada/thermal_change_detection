"""Run the pipeline on a rosbag without ROS running, for tuning.

    ros2 run thermal_change_detection offline_eval <bag> -o out/ \
        [--params config/params.yaml] [--image /flir_boson/image_raw] ...

Pass 1 loads all /tf and /tf_static into a TF buffer, pass 2 feeds the images.
Writes overlay_*.png (and reference_*.png when matched) plus log.csv to -o.
"""
import argparse
import csv
import os

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer, TransformException

from .core.config import PipelineConfig, load_ros_params_yaml
from .core.geometry import T_BODY_OPTICAL
from .core.pipeline import ChangeDetectionPipeline
from .node import tf_to_matrix


def open_reader(bag, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in topics if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    return reader, types


def add_bag_args(ap):
    """Arguments shared by the offline tools (bag, topics, frames, period)."""
    ap.add_argument('bag')
    ap.add_argument('--params', help='ROS params yaml (same format as config/params.yaml)')
    ap.add_argument('--image', default='/flir_boson/image_raw',
                    help='sensor_msgs/Image or CompressedImage topic')
    ap.add_argument('--camera-info', default='/flir_boson/camera_info')
    ap.add_argument('--map-frame', default='map')
    ap.add_argument('--camera-frame', default='', help='default: image header frame_id')
    ap.add_argument('--not-optical', action='store_true',
                    help='camera frame is x-forward/z-up instead of optical')
    ap.add_argument('--period', type=float, default=1.0)
    ap.add_argument('--max-frames', type=int, default=0)


def load_tf(bag):
    """Pass 1: all /tf and /tf_static of the bag into one TF buffer."""
    tf_buffer = Buffer(cache_time=Duration(seconds=10 ** 7))
    reader, types = open_reader(bag, ['/tf', '/tf_static'])
    n_tf = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        for tf in msg.transforms:
            if topic == '/tf_static':
                tf_buffer.set_transform_static(tf, 'bag')
            else:
                tf_buffer.set_transform(tf, 'bag')
            n_tf += 1
    print(f'  {n_tf} transforms')
    return tf_buffer


def iter_frames(args, tf_buffer):
    """Pass 2: yield (stamp_s, image, T_map_cam, camera_info) at most once per
    ``args.period``, the same subsampling as the node."""
    bridge = CvBridge()
    reader, types = open_reader(args.bag, [args.image, args.camera_info])
    if args.image not in types:
        raise SystemExit(f'topic {args.image} not in bag')
    compressed = types[args.image].endswith('CompressedImage')
    info, last_t = None, None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == args.camera_info:
            info = msg
            continue
        if info is None:
            continue
        stamp = Time.from_msg(msg.header.stamp)
        t = stamp.nanoseconds * 1e-9
        if last_t is not None and 0.0 <= t - last_t < args.period:
            continue
        frame = args.camera_frame or msg.header.frame_id
        try:
            T = tf_to_matrix(tf_buffer.lookup_transform(args.map_frame, frame, stamp))
        except TransformException as e:
            print(f'  skip {t:.2f}: {e}')
            continue
        if args.not_optical:
            T = T @ T_BODY_OPTICAL
        if compressed:
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        last_t = t
        yield t, img, T, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bag_args(ap)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--keep-keyframes', action='store_true')
    args = ap.parse_args()

    cfg = load_ros_params_yaml(args.params) if args.params else PipelineConfig()
    cfg.store.cleanup_on_exit = not args.keep_keyframes
    os.makedirs(args.out, exist_ok=True)

    print('pass 1: loading TF ...')
    tf_buffer = load_tf(args.bag)

    print('pass 2: processing images ...')
    pipeline = ChangeDetectionPipeline(cfg)
    n = 0
    log = open(os.path.join(args.out, 'log.csv'), 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['frame', 'stamp', 'keyframe', 'distance', 'angle_deg', 'correlation',
                     'aligned', 'gain', 'sigma', 'n_detections', 'inserted', 'status'])
    try:
        for t, img, T, info in iter_frames(args, tf_buffer):
            res = pipeline.process(img, t, T, np.array(info.k).reshape(3, 3),
                                   np.array(info.d), info.distortion_model)
            cv2.imwrite(os.path.join(args.out, f'overlay_{n:05d}.png'), res.overlay)
            if res.reference_view is not None:
                cv2.imwrite(os.path.join(args.out, f'reference_{n:05d}.png'), res.reference_view)
            m, a, c = res.match, res.alignment, res.change
            writer.writerow([
                n, f'{t:.3f}', m.keyframe.id if m else '', f'{m.distance:.3f}' if m else '',
                f'{m.angle_deg:.2f}' if m else '', f'{a.correlation:.3f}' if a else '',
                int(a.ok) if a else '', f'{c.gain:.4f}' if c else '', f'{c.sigma:.2f}' if c else '',
                len(res.detections), res.inserted.id if res.inserted else '', res.status])
            print(f'  [{n}] {res.status}')
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        log.close()
        pipeline.close()
    print(f'done: {n} frames -> {args.out}')


if __name__ == '__main__':
    main()
