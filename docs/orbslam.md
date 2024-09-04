# ORB_SLAM3

You can use ORB_SLAM3 as a backend in `stretch_ai`. It internally uses Stretch's D435i's RGBD and IMU (accelerometer and gyroscope) streams to build a map. `stretch_ros2_bridge` should automatically use ORB_SLAM's pose estimates and fuse them with Hector SLAM and wheel odometry. Please note that for ORB_SLAM3 to work reliably, it should always have sufficient texture and features visible in camera view.

We use our own version of [ORBSLAM3](https://github.com/hello-atharva/ORB_SLAM3) and associated [python bindings](https://github.com/hello-atharva/ORB_SLAM3-PythonBindings). The following steps will guide you through the installation process.

## Installation

To set up ORB_SLAM3 on your robot, execute the following commands in your terminal:

### Dependencies

Install build dependencies:

```
sudo apt install git cmake build-essential libboost-dev libssl-dev libboost-serialization-dev libboost-python-dev libeigen3-dev libgl1-mesa-dev libglew-dev libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran
```

### OpenCV4

Install OpenCV4

```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
mkdir -p ~/opencv_build/opencv/build && cd ~/opencv_build/opencv/build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules ..
make -j6 # This shall take a while
sudo make install
```

### Pangolin

Install Pangolin for visualization:

```
cd ~
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin/
sudo ./scripts/install_prerequisites.sh --dry-run recommended
cmake --build build
cd build
sudo make install
```

### Build and Install ORB_SLAM3

There are two build scripts that you can run on the robot as well:

- [ORBSLAM3 build script](https://github.com/hello-atharva/ORB_SLAM3/blob/main/build.sh)
- [ORBSLAM3 ROS build script](https://github.com/hello-atharva/ORB_SLAM3/blob/main/build_ros.sh)

Execute these commands:

```
cd ~
git clone https://github.com/hello-atharva/ORB_SLAM3.git
cd ORB_SLAM3/
./build.sh
cd build
sudo make install
cd ../Thirdparty/Sophus/build
sudo make install
```

This should also install Sophus on your system path.

### ORB_SLAM3 Python Wrapper

To use ORB_SLAM3 with Stretch AI, we use a custom python wrapper.

```
cd ~
git clone https://github.com/hello-atharva/ORB_SLAM3-PythonBindings.git
mkdir build
cd build
cmake ..
make j4
sudo make install
```

### Update LD_LIBRARY_PATH

This is to add previously installed libraries to the LD cache.

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```
