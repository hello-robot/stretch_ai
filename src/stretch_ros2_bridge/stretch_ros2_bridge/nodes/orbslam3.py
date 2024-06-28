#!/usr/bin/env python

import math
import os
import pathlib
import sys
from tempfile import NamedTemporaryFile
import time
import yaml

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import orbslam3
from stretch.navigation.utils.geometry import transformation_matrix_to_pose

class OrbSlam3(Node):
    def __init__(self):
        super().__init__('stretch_orbslam3')

        file_path = os.__file__
        DIRNAME = pathlib.Path(__file__).parent.resolve()
        PARENT_DIR = os.path.dirname(os.path.dirname(DIRNAME))
        CONFIG_FILE = os.path.join(PARENT_DIR, 'config', 'orbslam_d435i.yaml')
        self.VOCABULARY_FILE = os.path.join(PARENT_DIR, 'config', 'ORBvoc.txt')

        # Load YAML configuration
        self.config = None
        with open(CONFIG_FILE, 'r') as file:
            self.config = yaml.safe_load(file)

        self.slam = None

        self.tf_broadcaster = TransformBroadcaster(self)

        self.rgb_image = None
        self.depth_image = None
        self.accel_data = None
        self.gyro_data = None
        self.timestamp = None

        self.image_sub = self.create_subscription(Image,
                                                  "/camera/camera/color/image_raw",
                                                  self.rgb_callback,
                                                  1)
        self.depth_sub = self.create_subscription(Image,
                                                  "/camera/camera/aligned_depth_to_color/image_raw",
                                                  self.depth_callback,
                                                  1)
        self.accel_sub = self.create_subscription(Imu,
                                                  "/camera/camera/accel/sample",
                                                  qos_profile=rclpy.qos.qos_profile_sensor_data,
                                                  callback=self.accel_callback)
        self.gyro_sub = self.create_subscription(Imu,
                                                 "/camera/camera/gyro/sample",
                                                 qos_profile=rclpy.qos.qos_profile_sensor_data,
                                                 callback=self.gyro_callback)
        
        self.camera_info_sub = self.create_subscription(CameraInfo,
                                                        "/camera/camera/color/camera_info",
                                                        self.camera_info_callback,
                                                        1)

    def camera_info_callback(self, msg):
        if self.slam is None:
            fx = msg.K[0]
            fy = msg.K[4]
            cx = msg.K[2]
            cy = msg.K[5]

            # Modify camera intrinsics in config 
            self.config['Camera.fx'] = fx
            self.config['Camera.fy'] = fy
            self.config['Camera.cx'] = cx
            self.config['Camera.cy'] = cy

            height = msg.height
            width = msg.width

            # Modify image resolution in config
            self.config['Camera.height'] = height
            self.config['Camera.width'] = width

            # Save to a NamedTemporaryFile
            file = NamedTemporaryFile(mode='w', delete=False)
            yaml.dump(self.config, file)

            self.slam = orbslam3.System(self.VOCABULARY_FILE, file.name, orbslam3.Sensor.RGBD)
            self.slam.set_use_viewer(True)
            self.slam.initialize()
            print("ORB-SLAM3 initialized")

    def rgb_callback(self, msg):
        self.rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def depth_callback(self, msg):
        self.depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def accel_callback(self, msg):
        self.accel_data = msg

    def gyro_callback(self, msg):
        self.gyro_data = msg

    def run(self):
        while True:
            if self.rgb_image is None or \
                self.depth_image is None or \
                self.accel_data is None or \
                self.gyro_data is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            accel = self.accel_data.linear_acceleration
            gyro = self.gyro_data.angular_velocity

            Tcw = self.slam.process_image_rgbd(self.rgb_image,
                                               self.depth_image,
                                               [accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z],
                                               self.timestamp)
            
            Tcw = np.array(Tcw).reshape(4, 4)
            Twc = np.linalg.inv(Tcw)
            pose = transformation_matrix_to_pose(Twc)
            self.publish_tf(pose)

    def publish_tf(self, pose):
        x = pose.get_x()
        y = pose.get_y()
        z = pose.get_z()

        roll = pose.get_roll()
        pitch = pose.get_pitch()
        yaw = pose.get_yaw()
        q = quaternion_from_euler(yaw, pitch, -roll)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = z
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)

    node = OrbSlam3()
    node.run()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
