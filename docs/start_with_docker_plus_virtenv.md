# Install the Docker + Virtual Env Version of Stretch-AI

*stretch-ai* provides a [Docker](<https://en.wikipedia.org/wiki/Docker_(software)>) image and a [Python Virtual Environment](https://sciwiki.fredhutch.org/compdemos/python_virtual_environments/). The docker image is for your [Stretch](https://hello-robot.com/stretch-3-product) robot and the virtual env is for a computer with a GPU that communicates with your robot (*GPU computer*). This setup make it easier to develop on *stretch-ai*.

At the end of this docs, we also had a section describing how to launch your robot with traditional ROS2 installation.

Before you start trying docker, unplug the USB dongle on the robot.

## Install Docker on the Robot

Start by installing docker on your robot:

```
sudo apt-get update
sudo apt-get install docker.io
```

### Optional: Setup Docker Group So You Do Not Need To Use `sudo`

You can add your user to the `docker` group so you do not need to use `sudo` to run Docker commands. To do this, run the following command:

1. Create the docker group if it doesn't already exist:

```bash
sudo groupadd docker
```

2. Add your user to the docker group:

```bash
sudo usermod -aG docker $USER
```

3. Restart the Docker service:

```bash
sudo systemctl restart docker
```

4. Log out and log back in so that your group membership is re-evaluated. Then you can verify that you can run Docker commands without sudo:

```bash
docker run hello-world
```

If you want to run a docker command without logging out, you can run the following command:

```bash
newgrp docker
```

This will change your group to the `docker` group for the current terminal session.

**When performing these steps on your robot, you may find that logging out fails to make docker group membership take effect.** In this case, you can try restarting Docker with the following command:

```bash
sudo systemctl restart docker
```

If this doesn't work, we recommend that you reboot your robot's computer.


### Clone the Stretch-AI Repository on your robot

You will need to clone the *stretch-ai* repository on your robot to access the "robot script".

```bash
git clone https://github.com/hello-robot/stretch_ai.git
```

### Run the Robot's Script

The GitHub *stretch-ai* repository provides a startup script for running *stretch-ai* software in a Docker container on your Stretch robot. Prior to running the script, you need to have homed your robot with `stretch_robot_home.py`.

To use the Docker script, run the following command in the *stretch-ai* repository on the robot:

```
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

You will see something like the following in the terminal as the Docker image is downloaded:

```
(base) hello-robot@stretch-se3-2005:~/src/stretchpy$ ./scripts/run_stretch_ai_ros2_bridge_server.sh 
Starting Stretch AI ROS2 Bridge Server on stretch-se3-3005
=========================================================
Unable to find image 'hellorobotinc/stretch-ai-ros2-bridge:latest' locally
latest: Pulling from hellorobotinc/stretch-ai-ros2-bridge
762bedf4b1b7: Pull complete                                         
84ceaedb8a21: Pull complete                                         
c558ecc26f22: Pull complete                                         
1006c31c0071: Downloading [===>                                               ]  33.99MB/484MB
2883f1b72f50: Download complete                                     
c29b29edc871: Download complete                                     
75fa503deb0b: Download complete                                     
03297d3829eb: Download complete 
fcf26cd86178: Download complete 
5bcaaf1fd219: Download complete 
431ffe29be39: Download complete 
79e926b74f85: Download complete 
4f4fb700ef54: Verifying Checksum 
27ae57810c0a: Downloading [=>                                                 ]  19.43MB/570.5MB
9ecd20cd6844: Download complete 
51c071dfcd29: Download complete 
438302fc8bd8: Download complete 
44999d133959: Downloading [>                                                  ]  10.79MB/10.57GB
a5ed971e796e: Pulling fs layer                                      
f570a0dd636d: Waiting                                               
1a08cbb00ee1: Waiting                                               
```

The Docker image can be large (i.e., > 10GB), so it takes time to download. You can plug an ethernet cable into your router and Stretch to speed up the download. You will only need to download each version of the *stretch-ai* Docker image a single time.

After downloading the Docker image, the server will begin running on your robot. Your robot should beep twice and its lidar should start spinning. The terminal should display output from the robot's *stretch-ai* server like the following:

```bash
[server-12] j='joint_wrist_yaw' idx=10 idx_q=10
[server-12] j='joint_wrist_pitch' idx=11 idx_q=11
[server-12] j='joint_wrist_roll' idx=12 idx_q=12
[server-12] ==========================================
[server-12] Starting up threads:                                    
[server-12]  - Starting send thread
[server-12]  - Starting recv thread
[server-12]  - Sending state information
[server-12]  - Sending servo information
[server-12] Running all...                                          
[server-12] Starting to send full state
[stretch_driver-3] [INFO] [1727454898.725969113] [stretch_driver]: Changed to mode = position
```

## Create a virtual env on your GPU Computer

First, check to see if mamba is installed on your computer:

```bash
mamba
```

If you get an "command not found" error, then follow the install instructions for mamba here: https://github.com/conda-forge/miniforge#download

Make sure to run `mamba init` and restart your terminal before proceeding.

Then, run:

```bash
# Install Git LFS - needed for large files like images
sudo apt-get install git-lfs
git lfs install

# Clone the repository
# Do not forget the --recursive flag to clone submodules
git clone https://github.com/hello-robot/stretch_ai.git --recursive

# Run install script to create a conda environment and install dependencies
# WARNING: this will delete an existing stretch_ai environment - do not do this to update your code!
cd stretch_ai
./install.sh --cuda=$CUDA_VERSION --no-version
```

This will create a mamba environment called `stretch_ai` and install all the necessary dependencies.

Here, `$CUDA_VERSION` is the version of CUDA installed on your computer. If you don't know the version of CUDA installed on your computer, you can run the following command:

```bash
nvidia-smi
```

Or you can skip automatic perception installation as described in [Manual Perception Installation](#manual-perception-installation).

##### Updating Stretch AI

To update stretch AI, simply pull:
```
git pull -ff origin main

# Optional; rarely needed
git submodule update --init --recursive
```
*Do not run the install script again unless you want a new, clean environment.* Running the install script will delete your current environment. You can also run it without the `--no-version` flag to create a versioned environment, eg. `stretch_ai_0.1.16`:

```bash
./install.sh --cuda=$CUDA_VERSION
```

##### Optional: Use Conda instead of Mamba

You can pass the `--conda` flag into the install script if you have it installed:
```bash
./install.sh --conda
```

##### Manual Perception Installation

If you don't have CUDA installed or don't know what it is, you can answer **no** to the prompt to install Detic. If you do have CUDA installed, you can answer **yes** to the prompt to install Detic.

If you answered no, you can then install Detic manually. Take note of the name of the environment. It will be something like `stretch_ai_<version>`.

Next, run:

```bash
# Install detectron2 for perception (required by Detic)
git submodule update --init --recursive
cd third_party/detectron2
pip install -e .

# Install Detic for perception
cd ../../src/stretch/perception/detection/detic/Detic
# Make sure it's up to date
git submodule update --init --recursive
pip install -r requirements.txt

# Download DETIC checkpoint...
mkdir -p models
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

## Simple Installation Test

You can test your installation by running the `view_images` app.

While the Docker container on your robot is running, you can run the following commands in the terminal of your GPU computer.

First, you need to let the GPU computer know the IP address (#.#.#.#) for your Stretch robot.

```bash
./scripts/set_robot_ip.sh #.#.#.#
```

*Please note that it's important that your GPU computer and your Stretch robot be able to communicate via the following ports 4401, 4402, 4403, and 4404. If you're using a firewall, you'll need to open these ports.*

Next, run the `view_images` application in the Docker container on your GPU computer.

```
python -m stretch.app.view_images
```

With a functioning installation, the robot's gripper will open, the arm will move, and then you will see video from the robot's cameras displayed on your GPU computer.

To exit the app, you can press `q` with any of the popup windows selected.

If the `view_images` app doesn't work, the most common issue is that the GPU computer is unable to communicate with the robot over the network. We recommend that you verify your robot's IP address and use [`ping`](<https://en.wikipedia.org/wiki/Ping_(networking_utility)>) on your GPU computer to check that it can reach the robot.

If your installation is working, we recommend that you try out [language-directed pick and place](llm_agent.md).


# Installing ROS2 packages without docker
If you do not want to launch your robot with docker, you can also install ROS2 packages from ament workspace.
You first need to install Stretch AI environment with these commands:
```
# Ignore this line if you are already in stretch ai folder
cd stretch_ai/

bash ./install.sh
```

We assume [ROS2](https://docs.ros.org/en/foxy/index.html) has already been installed on your robot and the main directory for ros2 packages is `~/ament_ws`. 
After you install `stretch_ai`, you need to copy `stretch_ros2_bridge` folder in `stretch_ai/src/` folder into `~/ament_ws/` by
```
# Ignore this line if you are already in stretch ai folder
cd stretch_ai/

ln -s src/stretch_ros2_bridge ~/ament_ws/src/stretch_ros2_bridge
```
and run 
```
cd ~/ament_ws
colcon build --symlink
source install/setup.bash
```
to install ros2 package.