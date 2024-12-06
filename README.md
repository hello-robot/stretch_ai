# Stretch AI

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)

![PickPlaceFullTask](https://github.com/user-attachments/assets/a1db635c-03b5-48e8-9167-45f09bc8a9b2)

**Stretch AI** is designed to help researchers and developers build intelligent behaviors for the [Stretch 3](https://hello-robot.com/stretch-3-product) mobile manipulator from [Hello Robot](https://hello-robot.com/). It contains code for:

- grasping
- manipulation
- mapping
- navigation
- LLM agents
- text to speech and speech to text
- visualization and debugging

Much of the code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information. Parts of it are derived from the Meta [HomeRobot](https://github.com/facebookresearch/home-robot) project and are licensed under the [MIT license](META_LICENSE).

## Hardware Requirements

We recommend the following hardware to run Stretch AI. Other GPUs and other versions of Stretch may support some of the capabilities found in this repository, but our development and testing have focused on the following hardware.

- **[Stretch 3](https://hello-robot.com/stretch-3-product) from [Hello Robot](https://hello-robot.com/)**
  - When *Checking Hardware*, `stretch_system_check.py` should report that all hardware passes.
- **Computer with an NVIDIA GPU**
  - The computer should be running Ubuntu 22.04. Later versions might work, but have not been tested.
  - Most of our testing has used a high-end CPU with an NVIDIA GeForce RTX 4090.
- **Dedicated WiFi access point**
  - Performance depends on high-bandwidth, low-latency wireless communication between the robot and the GPU computer.
  - The official [Stretch WiFi Access Point](https://hello-robot.com/stretch-access-point) provides a tested example.
- (Optional) [Stretch Dexterous Teleop Kit](https://hello-robot.com/stretch-dex-teleop-kit).
  - To use the learning-from-demonstration (LfD) code you'll need the Stretch Dexterous Teleop Kit.

## Quick-start Guide

Artificial intelligence (AI) for robots often has complex dependencies, including the need for trained models. Consequently, installing *stretch-ai* from source can be challenging.

First, you will need to install software on your Stretch robot and another computer with a GPU (*GPU computer*). Use the following link to go to the installation instructions: [Instructions for Installing Stretch AI](https://github.com/hello-robot/stretch_ai/blob/main/docs/start_with_docker_plus_virtenv.md)

Once you've completed this installation, you can start the server on your Stretch robot.  Prior to running the script, you need to have homed your robot with `stretch_robot_home.py`. Then, run the following command:

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

After this, we recommend trying the [Language-Directed Pick and Place](#language-directed-pick-and-place) demo.

#### Experimental support for Older Robots

The older model of Stretch, the Stretch RE2, did not have an camera on the gripper. If you want to use this codebase with an older robot, you can purchase a [Stretch 2 Upgrade Kit](https://hello-robot.com/stretch-2-upgrade) to give your Stretch 2 the capabilities of a Stretch 3. Alternatively, you can run a version of the server with no d405 camera support on your robot.

Note that many demos will not work with this script (including the [Language-Directed Pick and Place](#language-directed-pick-and-place) demo) and [learning from demonstration](docs/learning_from_demonstration.md). However, you can still run the [simple motions demo](examples/simple_motions.py) and [view images](#visualization-and-streaming-video) with this script.

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh --no-d405
```

#### Optional: Docker Quickstart

To help you get started more quickly, we provide two pre-built [Docker](<https://en.wikipedia.org/wiki/Docker_(software)>) images that you can download and use with two shell scripts.

On your remote machine, you can install docker as normal, then, you can start the client on your GPU computer:

```bash
./scripts/run_stretch_ai_gpu_client.sh
```

This script will download the Docker image and start the container. You will be able to run Stretch AI applications from within the container.

### Language-Directed Pick and Place

![orangecupinbox](https://github.com/user-attachments/assets/f6659e40-8ed2-410a-889e-84f8bf8d38ad)

Now that you have the server running on Stretch, we recommend you try a demonstration of language-directed pick and place.

For this application, Stretch will attempt to pick up an object from the floor and place it inside a nearby receptacle on the floor. You will use words to describe the object and the receptacle that you'd like Stretch to use.

While attempting to perform this task, Stretch will speak to tell you what it is doing. So, it is a good idea to make sure that you have the speaker volume up on your robot. Both the physical knob on Stretch's head and the volume settings on Stretch's computer should be set so that you can hear what Stretch says.

Now, on your GPU computer, run the following commands in the Docker container that you started with the script above.

You need to let the GPU computer know the IP address (#.#.#.#) for your Stretch robot.

```bash
./scripts/set_robot_ip.sh #.#.#.#
```

*Please note that it's important that your GPU computer and your Stretch robot be able to communicate via the following ports 4401, 4402, 4403, and 4404. If you're using a firewall, you'll need to open these ports.*

Next, run the application on your GPU computer:

```bash
python -m stretch.app.ai_pickup
```

It will first spend time downloading various models that it depends on. Once the program starts, you will be able to bring up a [Rerun-based GUI](https://rerun.io/) in your web browser.

![Rerun-based GUI for the ai_pickup app.](docs/images/rerun_example.png)

Then, in the terminal, it will ask you to specify an object and a receptacle. For example, in the example pictured below, the user provided the following descriptions for the object and the receptacle.

```
Enter the target object: plush rabbit toy
Enter the target receptacle: yellow chair 
```

![RabbitChair](https://github.com/user-attachments/assets/dc7c19d2-49bd-45af-95ef-42abf22be5aa)

At Hello Robot, people have successfully commanded the robot to pick up a variety of objects from the floor and place them in nearby containers, such as baskets and boxes.

Find out more about the LLM-based AI agent in its [documentation](docs/llm_agent.md). And once you're ready to learn more about Stretch AI, you can try out the [variety of applications (apps)](docs/apps.md) that demonstrate various capabilities.

## Next Steps

Check out additional documentation for ways to use Stretch AI:

- [LLM Agent](docs/llm_agent.md) -- How to use the LLM agent for language-directed pick and place
- [Add a New LLM Task](docs/adding_a_new_task.md) -- How to add a new task that can be called by an LLM
- [DynaMem](docs/dynamem.md) -- Run the LLM agent in dynamic scenes, meaning you can walk around and place objects as the robot explores
- [Data Collection for Learning from Demonstration](docs/data_collection.md) -- How to collect data for learning from demonstration
- [Learning from Demonstration](docs/learning_from_demonstration.md)  -- How to train and evaluate policies with LfD
- [Open-Vocabulary Mobile Manipulation](docs/ovmm.md) -- Experimental code which can handle more complex language commands
- [Apps](docs/apps.md) -- List of many different apps that you can run
- [Simple API](docs/simple_api.md) -- How to use the simple API to control the robot over wireless

## Development

Clone this repo on your Stretch and PC, and install it locally using pip with the "editable" flag:

```
cd stretch_ai/src
pip install -e .[dev]
pre-commit install
```

Then follow the quickstart section. See [CONTRIBUTING.md](CONTRIBUTING.md) for more information. There is some information on how to [debug](docs/debug.md) and [update](docs/update.md) the codebase.

You can test out most code in the [simulation](docs/simulation.md) environment, which is a good way to test code without needing a robot.

### Updating Code on the Robot

See the [update guide](docs/update.md) for more information. Code installed from git must be updated manually, including code from this repository.

You can also pull the latest docker image on the robot with the following command:

```bash
./scripts/run_stretch_ai_ros2_bridge_server.sh --update
```

### Building Docker Images

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
