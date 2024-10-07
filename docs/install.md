# Stretch AI Installation

Stretch AI supports Python 3.10. We recommend using \[mamba\]https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to manage dependencies, or [starting with Docker](./start_with_docker.md).

If you do not start with docker, follow the [install guide](docs/install.md).

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

Even if you want to use the source install, we recommend [starting from Docker](docs/start_with_docker.md) on the robot at first.

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

## Installing CUDA 11.8

Make sure you have CUDA installed on your computer, preferably 11.8. It's possible to install multiple versions of CUDA on your computer, so make sure you have the correct version installed. You do not need to and should not install new versions of your NVIDIA drivers, but you may want to [install CUDA 11.8](https://developer.nvidia.com/cuda-11.8-download-archive) if you don't have it already, following the instructions in [Installing CUDA 11.8](#installing-cuda-11.8).

Download the runfile version of CUDA 11.8. E.g. for Ubuntu 22.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

![CUDA Installation](images/cuda_install.png)

Follow the prompts to install CUDA 11.8. When you get to the prompt to install the NVIDIA driver, select "No" to avoid installing a new driver. Also make sure you deselect the prompt for setting the system CUDA version!

Before running the install script, set the `$CUDA_HOME` environment variable to point to the new CUDA installation. For example, on Ubuntu 22.04:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
./install.sh
```

This should help avoid issues.

## Debugging Common Issues (Old)

First, verify that your installation is successful. One of the most common issues with the advanced installation is a CUDA version conflict, which means that Torch cannot run on your GPU.

### Verifying Torch GPU Installation

The most common issue is with `torch_cluster`, or that cuda is set up wrong. Make sure it runs by starting `python` and running:

```python
import torch_cluster
import torch
torch.cuda.is_available()
torch.rand(3, 3).to("cuda")
```

You should see:

- `torch_cluster` imports successfully
- `True` for `torch.cuda.is_available()`
- No errors for `torch.rand(3, 3).to("cuda")`

If instead you get an error, run the following to check your CUDA version:

```bash
nvcc --version
```

Note: if `nvcc --version` fails, try `/usr/local/cuda/bin/nvcc --version` instead.

Make sure you have CUDA installed on your computer, preferably 11.8. It's possible to install multiple versions of CUDA on your computer, so make sure you have the correct version installed. You do not need to and should not install new versions of your NVIDIA drivers, but you may want to [install CUDA 11.8](https://developer.nvidia.com/cuda-11.8-download-archive) if you don't have it already, following the instructions in [Installing CUDA 11.8](#installing-cuda-11.8).
