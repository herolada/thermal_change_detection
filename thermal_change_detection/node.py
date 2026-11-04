"""Thin ROS 2 wrapper around core.ChangeDetectionPipeline.

Subscribes:  image (sensor_msgs/Image, mono16 RAW16 preferred), camera_info
Uses TF:     map_frame <- camera optical frame at the image stamp
Publishes:   ~/overlay (bgr8), ~/diff (32FC1, signed, NaN outside overlap),
             ~/reference_warped (mono8), ~/detections (vision_msgs/Detection2DArray)
"""
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
from .core.geometry import T_BODY_OPTICAL, transform_to_matrix
from .core.pipeline import ChangeDetectionPipeline


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
        self.process_period = self.declare_parameter('process_period', 1.0).value
        # Max time [s, image-stamp time] an image may wait for its TF before
        # it is dropped. Waiting is non-blocking (retried on newer images).
        self.tf_timeout = self.declare_parameter('tf_timeout', 0.5).value

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
        self.last_stamp = None
        self.pending = None   # image due for processing, waiting for its TF

        sensor_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, 'camera_info', self.on_camera_info, sensor_qos)
        self.create_subscription(Image, 'image', self.on_image, sensor_qos)
        self.pub_overlay = self.create_publisher(Image, '~/overlay', 2)
        self.pub_diff = self.create_publisher(Image, '~/diff', 2)
        self.pub_ref = self.create_publisher(Image, '~/reference_warped', 2)
        self.pub_det = self.create_publisher(Detection2DArray, '~/detections', 10)

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_image(self, msg):
        info = self.camera_info
        if info is None:
            self.get_logger().warning('waiting for camera_info', throttle_duration_sec=5.0)
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.pending is None and not (
                self.last_stamp is not None and 0.0 <= t - self.last_stamp < self.process_period):
            self.pending = msg
        if self.pending is not None:
            self.try_process_pending(t, info)

    def try_process_pending(self, newest_t, info):
        """Process the pending image if its TF is available (never blocks)."""
        msg = self.pending
        stamp = Time.from_msg(msg.header.stamp)
        t = stamp.nanoseconds * 1e-9
        frame = self.camera_frame or msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, frame, stamp)
        except TransformException as e:
            if not 0.0 <= newest_t - t <= self.tf_timeout:
                self.pending = None
                self.get_logger().warning(
                    f'TF {self.map_frame} <- {frame} not available within '
                    f'{self.tf_timeout:.2f} s, dropping image: {e}', throttle_duration_sec=5.0)
            return
        self.pending = None
        T = tf_to_matrix(tf)
        if not self.frame_is_optical:
            T = T @ T_BODY_OPTICAL

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        K = np.array(info.k).reshape(3, 3)
        res = self.pipeline.process(img, t, T, K, np.array(info.d), info.distortion_model)
        self.last_stamp = t
        self.get_logger().debug(res.status)
        self.publish(res, msg.header)

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
