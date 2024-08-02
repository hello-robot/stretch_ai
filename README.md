# Stretch AI

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)

Tested with Python 3.9/3.10/3.11. **Development Notice**: The code in this repo is a work-in-progress. The code in this repo may be unstable, since we are actively conducting development. Since we have performed limited testing, you may encounter unexpected behaviors.

## Quickstart

After following the [installation instructions](#installation), start the server on your robot:

```bash
ros2 launch stretch_ros2_bridge server.launch.py
```

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
- [Test Timing](#test-timing) - Test the timing of the robot's control loop over the network.
- [Print Joint States](#print-joint-states) - Print the joint states of the robot.
- [View Images](#visualization-and-streaming-video) - View images from the robot's cameras.
- [Show Point Cloud](#show-point-cloud) - Show a joint point cloud from the end effector and head cameras.
- [Gripper](#gripper-tool) - Open and close the gripper.
- [Rerun](#rerun) - Start a [rerun.io](https://rerun.io/)-based web server to visualize data from your robot.

Advanced:

- [Automatic 3d Mapping](#automatic-3d-mapping) - Automatically explore and map a room, saving the result as a PKL file.
- [Read saved map](#voxel-map-visualization) - Read a saved map and visualize it.
- [Pickup Objects](#pickup-toys) - Have the robot pickup toys and put them in a box.

Debug tools:

- [Camera Info](#camera-info) - Print out camera information.

Finally:

- [Dex Teleop](#dex-teleop-for-data-collection) - Teleoperate the robot to collect demonstration data.

## Installation

On both your PC and your robot, clone and install the package:

```basha
git clone git@github.com:hello-robot/stretch_ai.git --recursive 
```

On your Stretch, symlink the `stretch_ros2_bridge` directory to your ament workspace and build:

```bash
cd stretch_ai
ln -s `pwd`/src/stretch_ros2_bridge $HOME/ament_ws/src/stretch_ros2_bridge
cd ~/ament_ws
colcon build --symlink-install --packages-select stretch_ros2_bridge
```

More instructions on the ROS2 bridge are in [its dedicated readme](src/stretch_ros2_bridge/README.md).

### Audio Dependencies

These are necessary for [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/), which is used for audio recording and playback. On Ubuntu, you can install them with:

```bash
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0

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

### Keyboard Teleop

Use the WASD keys to move the robot around.

```bash
python -m stretch.app.keyboard_teleop --robot_ip $ROBOT_IP

# You may also run in a headless mode without the OpenCV gui
python -m stretch.app.keyboard_teleop --headless
```

Remember, you should only need to provide the IP address the first time you run any app from a particular endpoint (e.g., your laptop).

### Test Timing

Test the timing of the robot's control loop over the network. This will print out the time it takes to send a command to the robot and receive a response. It will show a histogram after a fixed number of iterations given by the `--iterations` flag (default is 500).

```bash
python -m stretch.app.timing --robot_ip $ROBOT_IP

# Headless mode - no display
python -m stretch.app.timing --headless

# Set the number of iterations per histogram to 1000
python -m stretch.app.timing --iterations 1000
```

### Print Joint States

To make sure the robot is  connected or debug particular behaviors, you can print the joint states of the robot with the `print_joint_states` tool:

```bash
python -m stretch.app.print_joint_states --robot_ip $ROBOT_IP
```

You can also print out just one specific joint. For example, to just get arm extension in a loop, run:

```
python -m stretch.app.print_joint_states --joint arm
```

### Camera Info

Print out information about the cameras on the robot for debugging purposes:

```bash
python -m stretch.app.camera_info --robot_ip $ROBOT_IP
```

This will print out information about the resolutions of different images sent by the robot's cameras, and should show something like this:

```
---------------------- Camera Info ----------------------
Servo Head RGB shape: (320, 240, 3) Servo Head Depth shape: (320, 240)
Servo EE RGB shape: (240, 320, 3) Servo EE Depth shape: (240, 320)
Observation RGB shape: (640, 480, 3) Observation Depth shape: (640, 480)
```

### Visualization and Streaming Video

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

### Show Point Cloud

Show a joint point cloud from the end effector and head cameras. This will open an Open3d window with the point cloud, aggregated between the two cameras and displayed in world frame. It will additionally show the map's origin with a small coordinate axis; the blue arrow points up (z), the red arrow points right (x), and the green arrow points forward (y).

```bash
python -m stretch.app.show_point_cloud
```

You can use the `--reset` flag to put the robot into its default manipulation posture on the origin (0, 0, 0). Note that this is a blind, unsafe motion! Use with care.

```bash
python -m stretch.app.show_point_cloud --reset
```

### Gripper Tool

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

### Rerun Web Server

We provide the tools to publish information from the robot to a [Rerun](https://rerun.io/) web server. This is run automatically with our other apps, but if you want to just run the web server, you can do so with:

```bash
python -m stretch.app.rerun --robot_ip $ROBOT_IP
```

You should see something like this:

```
[2024-07-29T17:58:34Z INFO  re_ws_comms::server] Hosting a WebSocket server on ws://localhost:9877. You can connect to this with a native viewer (`rerun ws://localhost:9877`) or the web viewer (with `?url=ws://localhost:9877`).
[2024-07-29T17:58:34Z INFO  re_sdk::web_viewer] Hosting a web-viewer at http://localhost:9090?url=ws://localhost:9877
```

### Dex Teleop for Data Collection

Dex teleop is a low-cost system for providing user demonstrations of dexterous skills right on your Stretch. It has two components:

- `follower` runs on the robot, publishes video and state information, and receives goals from a large remote server
- `leader` runs on a GPU enabled desktop or laptop, where you can run a larger neural network.

To start it, on the robot, run:

```bash
python -m stretch.app.dex_teleop.follower
# You can run it in fast mode once you are comfortable with execution
python -m stretch.app.dex_teleop.follower --fast
```

On a remote, GPU-enabled laptop or workstation connected to the [dex telop setup](https://github.com/hello-robot/stretch_dex_teleop):

```bash
python -m stretch.app.dex_teleop.leader
```

[Read the Dex Teleop documentation](docs/dex_teleop.md) for more details.

### Automatic 3d Mapping

```bash
python -m stretch.app.mapping
```

You can show visualizations with:

```bash
python -m stretch.app.mapping --show-intermediate-maps --show-final-map
```

The flag `--show-intermediate-maps` shows the 3d map after each large motion (waypoint reached), and `--show-final-map` shows the final map after exploration is done.

It will record a PCD/PKL file which can be interpreted with the `read_sparse_voxel_map` script; see below.

Another useful flag when testing is the `--reset` flag, which will reset the robot to the starting position of (0, 0, 0). This is done blindly before any execution or mapping, so be careful!

### Voxel Map Visualization

You can test the voxel code on a captured pickle file:

```bash
python -m stretch.app.read_sparse_voxel_map -i ~/Downloads/stretch\ output\ 2024-03-21/stretch_output_2024-03-21_13-44-19.pkl
```

Optional open3d visualization of the scene:

```bash
python -m stretch.app.read_sparse_voxel_map -i ~/Downloads/stretch\ output\ 2024-03-21/stretch_output_2024-03-21_13-44-19.pkl  --show-svm
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
cd stretchpy/src
pip install -e .[dev]
pre-commit install
```

Then follow the quickstart section. See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

### Code Overview

The code is organized as follows. Inside the core package `src/stretch`:

- [core](src/stretch/core) is basic tools and interfaces
- [app](src/stretch/app)  contains individual endpoints, runnable as `python -m stretch.app.<app_name>`, such as mapping, discussed above.
- [motion](src/stretch/motion) contains motion planning tools, including [algorithms](src/stretch/motion/algo) like RRT.
- [mapping](src/stretch/mapping) is broken up into tools for voxel (3d / ok-robot style), instance mapping
- [agent](src/stretch/agent) is aggregate functionality, particularly robot_agent which includes lots of common tools including motion planning algorithms.
  - In particular, `agent/zmq_client.py` is specifically the robot control API, an implementation of the client in core/interfaces.py. there's another ROS client in `stretch_ros2_bridge`.
  - [agent/robot_agent.py](src/stretch/agent/robot_agent.py) is the main robot agent, which is a high-level interface to the robot. It is used in the `app` scripts.
  - [agent/base](src/stretch/agent/base) contains base classes for creating tasks, such as the [TaskManager](src/stretch/agent/base/task_manager.py) class and the [ManagedOperation](src/stretch/agent/base/managed_operation.py) class.
  - [agent/task](src/stretch/agent/task) contains task-specific code, such as for the `pickup` task. This is divided between "Managers" like [pickup_manager.py](src/stretch/agent/task/pickup_manager.py) which are composed of "Operations." Each operation is a composable state machine node with pre- and post-conditions.
  - [agent/operations](src/stretch/agent/operations) contains the individual operations, such as `move_to_pose.py` which moves the robot to a given pose.

The [stretch_ros2_bridge](src/stretch_ros2_bridge) package is a ROS2 bridge that allows the Stretch AI code to communicate with the ROS2 ecosystem. It is a separate package that is symlinked into the `ament_ws` workspace on the robot.

### Updating Code on the Robot

See the [update guide](docs/update.md) for more information. There is an [update script](scripts.update.sh) which should handle some aspects of this. Code installed from git must be updated manually, including code from this repository.

### Docker

Docker build and other instructions are located in the [docker guide](docs/docker.md). Generally speaking, from the root of the project, you  can run the docker build process with:

```
docker build -t stretch-ai_cuda-11.8:latest .
```

See the [docker guide](docs/docker.md) for more information and troubleshooting advice.
