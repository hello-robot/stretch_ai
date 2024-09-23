# Stretch AI

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)

*This repository is currently under active development and is subject to change.*

It is a pre-release codebase designed to enable developers to build intelligent behaviors on mobile robots in real homes. It contains code for:

- grasping
- manipulation
- mapping
- navigation
- LLM agents
- text to speech and speech to text
- visualization and debugging

This code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information. Parts of it are derived from the Meta [HomeRobot](https://github.com/facebookresearch/home-robot) project and are licensed under the [MIT license](META_LICENSE).

## Quickstart

After following the [installation instructions](#installation), start the server on your robot:

```bash
ros2 launch stretch_ros2_bridge server.launch.py
```

Make sure the core test app runs:

```python
python -m stretch.app.view_images --robot_ip $ROBOT_IP
```

You should see windows popping up with camera viers from the robot, and the arm should move into a default position. The head should also turn to face the robot hand. If this all happens, you are good to go! Press `q` to quit the app.

Then, on your PC, you can easily send commands and stream data:

```python
from stretch.agent import RobotClient
robot = RobotClient(robot_ip="192.168.1.15")  # Replace with your robot's IP
# On future connection attempts, the IP address can be left blank

# Turn head towards robot's hand
robot.move_to_manip_posture()

# Move forward 0.1 along robot x axis in maniplation mode, and move arm to 0.5 meter height
robot.arm_to([0.1, 0.5, 0, 0, 0, 0])

# Turn head towards robot's base and switch base to navigation mode
# In navigation mode, we can stream velocity commands to the base for smooth motions, and base
# rotations are enabled
robot.move_to_nav_posture()

# Move the robot back to origin
# navigate_to() is only allowed in navigation mode
robot.navigate_to([0, 0, 0])

# Move the robot 0.5m forward
robot.navigate_to([0.5, 0, 0], relative=True)

# Rotate the robot 90 degrees to the left
robot.navigate_to([0, 0, 3.14159/2], relative=True)

# And to the right
robot.navigate_to([0, 0, -3.14159/2], relative=True)
```

## Apps

After [installation](#installation), on the robot, run the server:

```bash
ros2 launch stretch_ros2_bridge server.launch.py
```

Then, first try these apps to make sure connections are working properly:

- [Keyboard Teleop](#keyboard-teleop) - Teleoperate the robot with the keyboard.
- [Print Joint States](#print-joint-states) - Print the joint states of the robot.
- [View Images](#visualization-and-streaming-video) - View images from the robot's cameras.
- [Show Point Cloud](#show-point-cloud) - Show a joint point cloud from the end effector and head cameras.
- [Gripper](#use-the-gripper) - Open and close the gripper.
- [Rerun](#rerun) - Start a [rerun.io](https://rerun.io/)-based web server to visualize data from your robot.
- [LLM Voice Chat](#voice-chat) - Chat with the robot using LLMs.

Advanced:

- [Automatic 3d Mapping](#automatic-3d-mapping) - Automatically explore and map a room, saving the result as a PKL file.
- [Read saved map](#voxel-map-visualization) - Read a saved map and visualize it.
- [Pickup Objects](#pickup-toys) - Have the robot pickup toys and put them in a box.

Finally:

- [Dex Teleop data collection](#dex-teleop-for-data-collection) - Dexterously teleoperate the robot to collect demonstration data.
- [Learning from Demonstration (LfD)](docs/learning_from_demonstration.md) - Train SOTA policies using [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

There are also some apps for [debugging](docs/debug.md).

## Installation

### System Dependencies

You need git-lfs:

```bash
sudo apt-get install git-lfs
git lfs install
```

You also need some system audio dependencies. These are necessary for [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/), which is used for audio recording and playback. On Ubuntu, you can install them with:

```bash
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg
```

### Install Stretch AI

On both your PC and your robot, clone and install the package:

```bash
git clone git@github.com:hello-robot/stretch_ai.git --recursive
```

#### Install On PC

The installation script will install the package and its dependencies, as well as (optionally) some perception modules.

```bash
cd stretch_ai
./install.sh
```

#### Install On the Robot

Robot installation can be tricky, because we use some features from [ROS2](https://docs.ros.org/en/humble/index.html), specifically the [Nav2](https://github.com/ros-navigation/navigation2) package for LIDAR slam.

You will need to link Stretch AI into your ROS workspace. There are two ways to do this; either install stretch AI in your base python environment, or link the conda environment into ROS (advanced). Either way, you will then need to [set up the ROS2 bridge](#set-up-ament-workspace) in your Ament workspace.

*Why all this complexity?* We run a set of ROS2 nodes based on the [HomeRobot](https://github.com/facebookresearch/home-robot) and [OK-Robot](https://ok-robot.github.io/) codebases for mobile manipulation and localization. In particular, this allows us to use [Nav2](https://docs.nav2.org/), a very well-tested ROS2 navigation stack, for localization, which makes it easier to build complex applications. You do not need to understand ROS2 to use this stack.

##### Option 1: Install Stretch AI in Base Python Environment

To install in the base python environment, you need to make sure build tools are up to date:

```bash
conda deactivate  # only if you are in a conda environment
pip install --upgrade pip setuptools packaging build meson ninja
```

This is particularly an issue for scikit-fmm, which is used for motion planning. After this is done, you can install the package as normal:

```bash
pip install ./src
```

Then, [set up the ROS2 bridge](#set-up-ament-workspace-on-the-robot).

##### Option 2: Link Conda Environment into ROS (Advanced).

If you are using a conda environment, you can link the conda environment into ROS. This is a bit more advanced, but can be useful if you want to keep your ROS and conda environments separate.

Install using the installation script, but using the `--cpu` flag for a CPU-only installation:

```bash
./install.sh --cpu
```

Then, activate the conda environment:

```bash
conda activate stretch_ai_$VERSION_cpu
```

Then, [link the package into your ament workspace](#set-up-ament-workspace-on-the-robot) and install the package:

```bash
colcon build --cmake-args -DPYTHON_EXECUTABLE=$(which python)
```

Some ROS python repositories might be missing - specifically `empy` and `catkin_pkg`. You can install these with:

```bash
python -m pip install empy catkin_pkg
```

#### Set Up Ament Workspace on the Robot

On your Stretch, symlink the `stretch_ros2_bridge` directory to your ament workspace and build:

```bash
cd stretch_ai
ln -s `pwd`/src/stretch_ros2_bridge $HOME/ament_ws/src/stretch_ros2_bridge
cd ~/ament_ws
colcon build --packages-select stretch_ros2_bridge
```

You need to rebuild the ROS2 bridge every time you update the codebase. You can do this with:

```bash
cd ~/ament_ws
colcon build --packages-select stretch_ros2_bridge
```

#### Experimental: Install ORB-SLAM3 On the Robot (Advanced)

[ORB-SLAM3](https://arxiv.org/pdf/2007.11898) is an open-source VSLAM (visual slam) library. Using it in conjunction with LIDAR-based localization can improve performance in many environments. Installation is documented in a [separate file](docs/orbslam3.md).

*Installation is not required to use Stretch AI.* If you chose to do so, you can then then use the ORB-SLAM3 version of the server launch file:

```
ros2 launch stretch_ros2_bridge server_orbslam3.launch.py
```

### Using LLMs

We use many open-source LLMs from [Huggingface](https://huggingface.co/). TO use them, you will need to make sure `transformers` is installed and up to date. You can install it with:

```bash
pip install transformers --upgrade
```

You will need to go to the associated websites and accept their license agreements.

- [Gemma 2](https://huggingface.co/google/gemma-2b)
- [Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

Then you need to login to the huggingface CLI:

```bash
huggingface-cli login
```

This will require a personal access token created on the Huggingface website. After this, you can test LLM chat APIs via:

```bash
# Start a local chat with Gamma 2-2B -- requires ~5gb GPU memory
python -m stretch.llms.gemma_client

# Start a local chat with Llama 3.1 8B -- requires a bigger GPU
python -m stretch.llms.llama_client
```

### Using OVMM

You can use the LLMs above to plan long-horizon tasks by generating code as robot policies:

```bash
python -m stretch.app.ovmm
```

The robot will start exploring its workspace and will soon ask you for a text command.

### Advanced Installation (PC Only)

If you want to install AI code using pytorch, run the following on your GPU-enabled workstation:

```
./install.sh
```

Caution, it may take a while! Several libraries are built from source to avoid potential compatibility issues.

You may need to configure some options for the right pytorch/cuda version. Make sure you have CUDA installed on your computer, preferably 11.8. For issues, see [docs/about_advanced_installation.md](docs/about_advanced_installation.md).

## Stretch AI Apps

Stretch AI is a collection of tools and applications for the Stretch robot. These tools are designed to be run on the robot itself, or on a remote computer connected to the robot. The tools are designed to be run from the command line, and are organized as Python modules. You can run them with `python -m stretch.app.<app_name>`.

Some, like `print_joint_states`, are simple tools that print out information about the robot. Others, like `mapping`, are more complex and involve the robot moving around and interacting with its environment.

All of these take the `--robot_ip` flag to specify the robot's IP address. You should only need to do this the first time you run an app for a particular IP address; the app will save the IP address in a configuration file at `~/.stretch/robot_ip.txt`. For example:

```bash
export ROBOT_IP=192.168.1.15
python -m stretch.app.print_joint_states --robot_ip $ROBOT_IP
```

### Debugging Tools

#### Keyboard Teleop

Use the WASD keys to move the robot around.

```bash
python -m stretch.app.keyboard_teleop --robot_ip $ROBOT_IP

# You may also run in a headless mode without the OpenCV gui
python -m stretch.app.keyboard_teleop --headless
```

Remember, you should only need to provide the IP address the first time you run any app from a particular endpoint (e.g., your laptop).

#### Print Joint States

To make sure the robot is connected or debug particular behaviors, you can print the joint states of the robot with the `print_joint_states` tool:

```bash
python -m stretch.app.print_joint_states --robot_ip $ROBOT_IP
```

You can also print out just one specific joint. For example, to just get arm extension in a loop, run:

```
python -m stretch.app.print_joint_states --joint arm
```

#### Visualization and Streaming Video

Visualize output from the caneras and other sensors on the robot. This will open multiple windows with wrist camera and both low and high resolution head camera feeds.

```bash
python -m stretch.app.view_images --robot_ip $ROBOT_IP
```

You can also visualize it with semantic segmentation (defaults to [Detic](https://github.com/facebookresearch/Detic/):

```bash
python -m stretch.app.view_images --robot_ip $ROBOT_IP ----run_semantic_segmentation
```

You can visualize gripper Aruco markers as well; the aruco markers can be used to determine the finger locations in the image.

```bash
python -m stretch.app.view_images --robot_ip $ROBOT_IP --aruco
```

#### Show Point Cloud

Show a joint point cloud from the end effector and head cameras. This will open an Open3d window with the point cloud, aggregated between the two cameras and displayed in world frame. It will additionally show the map's origin with a small coordinate axis; the blue arrow points up (z), the red arrow points right (x), and the green arrow points forward (y).

```bash
python -m stretch.app.show_point_cloud
```

You can use the `--reset` flag to put the robot into its default manipulation posture on the origin (0, 0, 0). Note that this is a blind, unsafe motion! Use with care.

```bash
python -m stretch.app.show_point_cloud --reset
```

#### Use the Gripper

Open and close the gripper:

```
python -m stretch.app.gripper --robot_ip $ROBOT_IP --open
python -m stretch.app.gripper --robot_ip $ROBOT_IP --close
```

Alternately:

```
python -m stretch.app.open_gripper --robot_ip $ROBOT_IP
python -m stretch.app.close_gripper --robot_ip $ROBOT_IP
```

#### Rerun Web Server

We provide the tools to publish information from the robot to a [Rerun](https://rerun.io/) web server. This is run automatically with our other apps, but if you want to just run the web server, you can do so with:

```bash
python -m stretch.app.rerun --robot_ip $ROBOT_IP
```

You should see something like this:

```
[2024-07-29T17:58:34Z INFO  re_ws_comms::server] Hosting a WebSocket server on ws://localhost:9877. You can connect to this with a native viewer (`rerun ws://localhost:9877`) or the web viewer (with `?url=ws://localhost:9877`).
[2024-07-29T17:58:34Z INFO  re_sdk::web_viewer] Hosting a web-viewer at http://localhost:9090?url=ws://localhost:9877
```

### Voice Chat

Chat with the robot using LLMs.

```bash
python -m stretch.app.voice_chat
```

### Dex Teleop for Data Collection

Dex teleop is a low-cost system for providing user demonstrations of dexterous skills right on your Stretch. This app requires the use of the [dex teleop kit](https://hello-robot.com/stretch-dex-teleop-kit).

You need to install mediapipe for hand tracking:

```
python -m pip install mediapipe
```

```bash
python -m stretch.app.dex_teleop.ros2_leader -i $ROBOT_IP --teleop-mode base_x --save-images --record-success --task-name default_task
```

[Read the data collection documentation](docs/data_collection.md) for more details.

After this, [read the learning from demonstration instructions](docs/learning_from_demonstration.md) to train a policy.

### Automatic 3d Mapping

```bash
python -m stretch.app.mapping
```

You can show visualizations with:

```bash
python -m stretch.app.mapping --show-intermediate-maps --show-final-map
```

The flag `--show-intermediate-maps` shows the 3d map after each large motion (waypoint reached), and `--show-final-map` shows the final map after exploration is done.

It will record a PCD/PKL file which can be interpreted with the `read_map` script; see below.

Another useful flag when testing is the `--reset` flag, which will reset the robot to the starting position of (0, 0, 0). This is done blindly before any execution or mapping, so be careful!

### Voxel Map Visualization

You can test the voxel code on a captured pickle file. We recommend trying with the included [hq_small.pkl](src/test/mapping/hq_small.pkl)  or [hq_large](src/test/mapping/hq_large.pkl) files, which contain a short and a long captured trajectory from Hello Robot.

```bash
python -m stretch.app.read_map -i hq_small.pkl
```

Optional open3d visualization of the scene:

```bash
python -m stretch.app.read_map -i hq_small.pkl  --show-svm
```

You can visualize instances in the voxel map with the `--show-instances` flag:

```bash
python -m stretch.app.read_map -i hq_small.pkl  --show-instances
```

You can also re-run perception with the `--run-segmentation` flag and provide a new export file with the `--export` flag:

```bash
 python -m stretch.app.read_map -i hq_small.pkl --export hq_small_v2.pkl --run-segmentation
```

You can test motion planning, frontier exploration, etc., as well. Use the `--start` flag to set the robot's starting position:

```bash
# Test motion planning
python -m stretch.app.read_map -i hq_small.pkl --test-planning --start 4.5,1.3,2.1
# Test planning to frontiers with current parameters file
python -m stretch.app.read_map -i hq_small.pkl --test-plan-to-frontier --start 4.0,1.4,0.0
# Test sampling movement to objects
python -m stretch.app.read_map -i hq_small.pkl --test-sampling --start 4.5,1.4,0.0
# Test removing an object from the map
python -m stretch.app.read_map -i hq_small.pkl --test-remove --show-instances --query "cardboard box"
```

### Pickup Objects

This will have the robot move around the room, explore, and pickup toys in order to put them in a box.

```bash
python -m stretch.app.pickup --target_object toy
```

You can add the `--reset` flag to make it go back to the start position. The default object is "toy", but you can specify other objects as well, like "bottle", "cup", or "shoe".

```
python -m stretch.app.pickup --reset
```

## Development

Clone this repo on your Stretch and PC, and install it locally using pip with the "editable" flag:

```
cd stretch_ai/src
pip install -e .[dev]
pre-commit install
```

Then follow the quickstart section. See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

### Updating Code on the Robot

See the [update guide](docs/update.md) for more information. There is an [update script](scripts.update.sh) which should handle some aspects of this. Code installed from git must be updated manually, including code from this repository.

### Docker

Docker build and other instructions are located in the [docker guide](docs/docker.md). Generally speaking, from the root of the project, you can run the docker build process with:

```
docker build -t stretch-ai_cuda-11.8:latest .
```

See the [docker guide](docs/docker.md) for more information and troubleshooting advice.

## Acknowledgements

Parts of this codebase were derived from the Meta [HomeRobot](https://github.com/facebookresearch/home-robot) project, and is licensed under the [MIT license](META_LICENSE). We thank the Meta team for their contributions.

The [stretch_ros2_bridge](src/stretch_ros2_bridge) package is based on the [OK robot](https://github.com/ok-robot/ok-robot) project's [Robot Controller](https://github.com/NYU-robot-learning/robot-controller/), and is licensed under the [Apache 2.0 license](src/stretch_ros2_bridge/LICENSE).

We use [LeRobot from HuggingFace](https://github.com/huggingface/lerobot) for imitation learning, though we use [our own fork](https://github.com/hello-robot/lerobot).

## License

This code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information.
