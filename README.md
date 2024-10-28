# Stretch AI

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)

*This repository is currently under active development and is subject to change.*

**stretch-ai** is designed to help researchers and developers build intelligent behaviors for the [Stretch 3](https://hello-robot.com/stretch-3-product) mobile manipulator from [Hello Robot](https://hello-robot.com/). It contains code for

- grasping
- manipulation
- mapping
- navigation
- LLM agents
- text to speech and speech to text
- visualization and debugging

Much of the code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information. Parts of it are derived from the Meta [HomeRobot](https://github.com/facebookresearch/home-robot) project and are licensed under the [MIT license](META_LICENSE).

## Quick-start Guide

Artificial intelligence (AI) for robots often has complex dependencies, including the need for trained models. Consequently, installing **stretch-ai** from source can be challenging.

To help you get started more quickly, we provide two pre-built [Docker](<https://en.wikipedia.org/wiki/Docker_(software)>) images that you can download and use with two shell scripts.

First, you will need to install software on your Stretch robot and another computer with a GPU (*GPU computer*). Use the following link to go to the installation instructions.

[Instructions for Installing the Docker Version of stretch-ai](https://github.com/hello-robot/stretch_ai/blob/main/docs/start_with_docker.md)

Once you've completed this installation, you can start the server on your Stretch robot.

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

Then, you can start the client on your GPU computer.

```bash
./scripts/run_stretch_ai_gpu_client.sh
```

### Language-Directed Pick and Place

Now that you have the server running on Stretch and the client running on your GPU computer, we recommend you try a demonstration of language-directed pick and place.

For this application, Stretch will attempt to pick up an object from the floor and place it inside a nearby receptacle on the floor. You will use words to describe the object and the receptacle that you'd like Stretch to use.

While attempting to perform this task, Stretch will speak to tell you what it is doing. So, it is a good idea to make sure that you have the speaker volume up on your robot. Both the physical knob on Stretch's head and the volume settings on Stretch's computer should be set so that you can hear what Stretch says.

Now, on your GPU computer, run the following commands in the Docker container that you started with the script above.

You need to let the GPU computer know the IP address (#.#.#.#) for your Stretch robot.

```bash
./scripts/set_robot_ip.sh #.#.#.#
```

*Please note that it's important that your GPU computer and your Stretch robot be able to communicate via the following ports 4401, 4402, 4403, and 4404. If you're using a firewall, you'll need to open these ports.*

Next, run the application in the Docker container on your GPU computer.

```bash
python -m stretch.app.ai_pickup
```

It will first spend time downloading various models that it depends on. Then, it will ask you to specify an object and a receptacle. For example, in the example pictured below, the user provided the following descriptions for the object and the receptacle.

```bash
Enter the target object: brown moose toy
Enter the target receptacle: white laundry basket 
```

![Example of using the ai_pickup app with a toy moose and a laundry basket.](./docs/images/ai_pickup_moose_and_basket_example.jpg)

At Hello Robot, people have successfully commanded the robot to pick up a variety of objects from the floor and place them in nearby containers, such as baskets and boxes.

Once you're ready to learn more about **stretch-ai**, you can try out a variety of applications (apps) that demonstrate various capabilities.

## Apps

A large number of [apps](docs/apps.md) are available for the Stretch robot, providing various features and an easy way to get started using the robot.

After [installation](#installation), on the robot, run the server:

```bash
# If you did a manual install
ros2 launch stretch_ros2_bridge server.launch.py

# Alternately, via Docker -- will be slow the first time when image is downloaded!
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

Then, try the `view_images` app to make sure connections are working properly:

- [View Images](#visualization-and-streaming-video) - View images from the robot's cameras.

Next you can run the AI demo:

- [Pickup Objects](#pickup-toys) - Have the robot pickup toys and put them in a box.

Finally:

- [Automatic 3d Mapping](#automatic-3d-mapping) - Automatically explore and map a room, saving the result as a PKL file.
- [Read saved map](#voxel-map-visualization) - Read a saved map and visualize it.
- [Dex Teleop data collection](#dex-teleop-for-data-collection) - Dexterously teleoperate the robot to collect demonstration data.
- [Learning from Demonstration (LfD)](docs/learning_from_demonstration.md) - Train SOTA policies using [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

See the [apps documentation](docs/apps.md) for a complete list. There are also some apps for [debugging](docs/debug.md).

## Installation

Stretch AI supports Python 3.10. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to manage dependencies, or [starting with Docker](docs/start_with_docker.md).

If you do not start with Docker, follow the [install guide](docs/install.md).

In short, on the PC you will:

```bash
# Install Git LFS
sudo apt-get install git-lfs
git lfs install

# Clone the repository
git clone git@github.com:hello-robot/stretch_ai.git --recursive

# Install system dependencies
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg

# Run install script to create a conda environment and install dependencies
./install.sh
```

## Stretch AI Apps

Stretch AI is a collection of tools and applications for the Stretch robot. These tools are designed to be run on the robot itself, or on a remote computer connected to the robot. The tools are designed to be run from the command line, and are organized as Python modules. You can run them with `python -m stretch.app.<app_name>`.

Some, like `print_joint_states`, are simple tools that print out information about the robot. Others, like `mapping`, are more complex and involve the robot moving around and interacting with its environment.

All of these take the `--robot_ip` flag to specify the robot's IP address. You should only need to do this the first time you run an app for a particular IP address; the app will save the IP address in a configuration file at `~/.stretch/robot_ip.txt`. For example:

```bash
export ROBOT_IP=192.168.1.15
python -m stretch.app.print_joint_states --robot_ip $ROBOT_IP
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
