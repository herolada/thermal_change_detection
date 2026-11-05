"""Thin ROS 2 wrapper around core.ChangeDetectionPipeline.

Subscribes:  image (sensor_msgs/Image, mono16 RAW16 preferred), camera_info,
             optionally heatmap_topic (sensor_msgs/Image, false-colour view of
             the same camera: ~/overlay is then drawn on it)
Uses TF:     map_frame <- camera optical frame at the stamp of every image
             (cheap, ~0.2 ms); core.frame_select picks the images to process
Publishes:   ~/overlay (bgr8), ~/diff (32FC1, signed, NaN outside overlap),
             ~/reference_warped (mono8), ~/detections (vision_msgs/Detection2DArray)
"""
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .core.config import PipelineConfig, flatten, update_from_flat
from .core.frame_select import FrameSelector
from .core.geometry import T_BODY_OPTICAL, transform_to_matrix
from .core.pipeline import ChangeDetectionPipeline
from .core.stability import FrameBuffer


def tf_to_matrix(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


class ThermalChangeDetectionNode(Node):
    def __init__(self):
        super().__init__('thermal_change_detection')
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        # Empty -> use the image header frame_id.
        self.camera_frame = self.declare_parameter('camera_frame', '').value
        # False if camera_frame follows the body convention (x fwd, z up)
        # instead of the optical one (z fwd, y down).
        self.frame_is_optical = self.declare_parameter('frame_is_optical', True).value
        # Max time [s, image-stamp time] an image may wait for its TF before
        # it is dropped. Waiting is non-blocking (retried on newer images).
        self.tf_timeout = self.declare_parameter('tf_timeout', 0.5).value
        # Optional false-colour image of the same camera (original geometry,
        # same aspect ratio) that ~/overlay is drawn on. Empty -> off.
        self.heatmap_topic = self.declare_parameter('heatmap_topic', '').value
        self.heatmap_max_dt = self.declare_parameter('heatmap_max_dt', 0.02).value

        cfg = PipelineConfig()
        defaults = flatten(cfg)
        update_from_flat(cfg, {k: self.declare_parameter(k, v).value
                               for k, v in defaults.items()})
        self.pipeline = ChangeDetectionPipeline(cfg)
        self.get_logger().info(f'keyframes stored in {self.pipeline.store.run_dir}')

        self.tf_buffer = Buffer()
        # No spin_thread: it moves this whole node onto the listener's executor
        # (a node has one executor). Callbacks never block, so one thread is fine.
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.camera_info = None
        self.selector = FrameSelector(cfg.select, self.pipeline.store)
        self.unposed = deque()   # (stamp, msg) waiting for their TF, in order
        self.heatmaps = deque()  # (stamp, msg), paired with images before selection
        st, sel = cfg.stability, cfg.select
        # Recent raw images for the stability check (TF looked up only when
        # used). The selector may hold a frame for up to max(period, max_period).
        hold = max(sel.period, sel.max_period) if sel.mode == 'closest' else 0.0
        self.recent = FrameBuffer(hold + 2.0 * st.lookback_s + self.tf_timeout + 0.5) \
            if st.enabled else None

        sensor_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, 'camera_info', self.on_camera_info, sensor_qos)
        self.create_subscription(Image, 'image', self.on_image, sensor_qos)
        if self.heatmap_topic:
            self.create_subscription(Image, self.heatmap_topic, self.on_heatmap,
                                     QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                                                reliability=ReliabilityPolicy.BEST_EFFORT))
            self.get_logger().info(f'overlay drawn on {self.heatmap_topic}')
        self.pub_overlay = self.create_publisher(Image, '~/overlay', 2)
        self.pub_diff = self.create_publisher(Image, '~/diff', 2)
        self.pub_ref = self.create_publisher(Image, '~/reference_warped', 2)
        self.pub_det = self.create_publisher(Detection2DArray, '~/detections', 10)

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_heatmap(self, msg):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.heatmaps.append((t, msg))
        while t - self.heatmaps[0][0] > self.tf_timeout + 1.0:   # camera image stalled
            self.heatmaps.popleft()

    def pair_heatmap(self, t, now):
        """Heatmap msg closest to stamp t (within heatmap_max_dt), None if
        there is none, or False if it may still arrive (wait)."""
        q, tol = self.heatmaps, self.heatmap_max_dt
        while q and q[0][0] < t - tol:              # too old for t and every later image
            q.popleft()
        best = min(((abs(s - t), m) for s, m in q if abs(s - t) <= tol),
                   key=lambda e: e[0], default=None)
        if best is not None:
            return best[1]
        if (q and q[-1][0] > t + tol) or now - t > self.tf_timeout:
            self.get_logger().warning(f'no heatmap within {tol:.3f} s of image, '
                                      'drawing on the camera image', throttle_duration_sec=5.0)
            return None
        return False

    def on_image(self, msg):
        info = self.camera_info
        if info is None:
            self.get_logger().warning('waiting for camera_info', throttle_duration_sec=5.0)
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.recent is not None:
            self.recent.push(t, msg)
        self.unposed.append((t, msg))
        # Feed every image whose TF is available to the selector, in order
        # (never blocks: images still waiting for TF are retried next time).
        while self.unposed:
            ft, fmsg = self.unposed[0]
            try:
                T = self.lookup_pose(fmsg)
            except TransformException as e:
                if 0.0 <= t - ft <= self.tf_timeout:
                    break
                self.unposed.popleft()
                self.get_logger().warning(
                    f'TF {self.map_frame} <- {self.camera_frame or fmsg.header.frame_id} not '
                    f'available within {self.tf_timeout:.2f} s, dropping image: {e}',
                    throttle_duration_sec=5.0)
                continue
            heat = None
            if self.heatmap_topic:
                heat = self.pair_heatmap(ft, t)
                if heat is False:                   # not received yet: retry next image
                    break
            self.unposed.popleft()
            sel = self.selector.push(ft, T, (fmsg, heat))
            if sel is not None:
                self.process(sel, info)

    def lookup_pose(self, msg):
        """T_map_cam (optical) at the image stamp; raises TransformException."""
        tf = self.tf_buffer.lookup_transform(
            self.map_frame, self.camera_frame or msg.header.frame_id,
            Time.from_msg(msg.header.stamp))
        T = tf_to_matrix(tf)
        return T if self.frame_is_optical else T @ T_BODY_OPTICAL

    def process(self, sel, info):
        msg, heat_msg = sel.item
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        heat = None if heat_msg is None else \
            self.bridge.imgmsg_to_cv2(heat_msg, desired_encoding='bgr8')
        K = np.array(info.k).reshape(3, 3)
        res = self.pipeline.process(img, sel.stamp, sel.T_map_cam, K, np.array(info.d),
                                    info.distortion_model, aux=self.aux_frame(sel.stamp),
                                    heatmap=heat)
        self.get_logger().debug(f'{res.status} [{sel.reason}]')
        self.publish(res, msg.header)

    def aux_frame(self, t):
        """(image, stamp, T_map_cam) of the buffered frame closest to
        stability.lookback_s before t, or None."""
        if self.recent is None:
            return None
        picked = self.recent.pick(t, self.pipeline.cfg.stability.lookback_s)
        if picked is None:
            return None
        a_t, msg = picked
        try:
            T = self.lookup_pose(msg)
        except TransformException:
            return None
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'), a_t, T

    def publish(self, res, header):
        out = self.bridge.cv2_to_imgmsg(res.overlay, encoding='bgr8')
        out.header = header
        self.pub_overlay.publish(out)
        if res.change is not None:
            out = self.bridge.cv2_to_imgmsg(res.change.diff, encoding='32FC1')
            out.header = header
            self.pub_diff.publish(out)
            out = self.bridge.cv2_to_imgmsg(res.reference_view, encoding='mono8')
            out.header = header
            self.pub_ref.publish(out)

        arr = Detection2DArray(header=header)
        for i, b in enumerate(res.detections):
            d = Detection2D(header=header, id=f'{b.kind}_{i}')
            x, y, w, h = b.bbox
            d.bbox.center.position.x = x + w / 2.0
            d.bbox.center.position.y = y + h / 2.0
            d.bbox.size_x, d.bbox.size_y = float(w), float(h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = b.kind
            # change blobs: peak |z| (noise-sigma units); hotspots: peak intensity
            hyp.hypothesis.score = float(b.peak_z if b.kind != 'hotspot' else b.peak_value)
            d.results.append(hyp)
            arr.detections.append(d)
        self.pub_det.publish(arr)

    def destroy_node(self):
        self.pipeline.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ThermalChangeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
