import os
from glob import glob

from setuptools import find_packages, setup

package_name = "robot_hw_python"

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
    ],
    install_requires=["setuptools", "home_robot"],
    zip_safe=True,
    maintainer="hello-robot",
    maintainer_email="hello-robot@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_pose_publisher = robot_hw_python.nodes.camera_pose_publisher:main",
            "rotate_images = scripts.rotate_images",
            "state_estimator = robot_hw_python.nodes.state_estimator:main",
            "goto_controller = robot_hw_python.nodes.goto_controller:main",
            "odom_tf_publisher = robot_hw_python.nodes.odom_tf_publisher:main",
            "server = robot_hw_python.remote.server:main",
        ],
    },
)
