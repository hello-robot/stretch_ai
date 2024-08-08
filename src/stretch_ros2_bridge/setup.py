# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "stretch_ros2_bridge"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*")),
        ),
    ],
    install_requires=["setuptools", "stretch"],
    zip_safe=True,
    maintainer="hello-robot",
    maintainer_email="hello-robot@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_pose_publisher = stretch_ros2_bridge.nodes.camera_pose_publisher:main",
            "rotate_images = scripts.rotate_images",
            "state_estimator = stretch_ros2_bridge.nodes.state_estimator:main",
            "goto_controller = stretch_ros2_bridge.nodes.goto_controller:main",
            "odom_tf_publisher = stretch_ros2_bridge.nodes.odom_tf_publisher:main",
            "orbslam3 = stretch_ros2_bridge.nodes.orbslam3:main",
            "server = stretch_ros2_bridge.remote.server:main",
        ],
    },
)
