# Install the Docker Version of Stretch-AI

*stretch-ai* provides two [Docker](<https://en.wikipedia.org/wiki/Docker_(software)>) images, one for your [Stretch](https://hello-robot.com/stretch-3-product) robot and one for a computer with a GPU that communicates with your robot (*GPU computer*). These two Docker images make it easier to try *stretch-ai*.

The Docker images enables you to run *stretch-ai* in [containers](<https://en.wikipedia.org/wiki/Containerization_(computing)>) that include *stretch-ai* and its dependencies. Using a container simplifies installation and reduces the likelihood of interfering with other uses of your robot and your GPU computer.

**Please note that changes to the Docker container's files and the Docker container's state will be lost once you stop the container.**

## Install Docker

Start by installing docker on your robot and your GPU computer:

```
sudo apt-get update
sudo apt-get install docker.io
```

On the GPU computer you also need the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Check the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the most up-to-date instructions.

You can also install with:

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Then restart the Docker service:

```
sudo systemctl restart docker
```

An [nvidia docker install script](scripts/install_nvidia_container_toolkit.sh) has been provided for Ubuntu machines. Again, though, [check the official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the most up-to-date instructions and if you have any issues.

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

## Clone the Stretch-AI Repository

You will need to clone the *stretch-ai* repository on your robot and on your GPU computer. For example, you could run the following command on your robot and on your GPU computer.

```
git clone https://github.com/hello-robot/stretch_ai.git
```

## Run the Robot's Script

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

## Run the GPU Computer's Script

Now you need to start the container on your GPU computer. To use the script, run the following command in the *stretch-ai* repository on the robot:

```bash
./scripts/run_stretch_ai_gpu_client.sh
```

#### Verify NVIDIA Docker

Make sure NVIDIA version of Docker is set up correctly. To do this, run the `nvidia-smi` command in the docker shell.

You should see something like the following:

```bash
root@olympia:/app# nvidia-smi
Fri Sep 27 16:12:43 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:01:00.0 Off |                  Off |
|  0%   38C    P8             22W /  450W |     415MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

#### Use the Mamba Environment

```bash
# Activate the environment
mamba init && source ~/.bashrc && mamba activate stretch_ai
```

## Simple Installation Test

You can test your Docker installation by running the `view_images` app.

While the Docker container on your robot is running, you can run the following commands in the terminal for your GPU computer's Docker container.

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

If your installation is working, we recommend that you try out [language-directed pick and place](https://github.com/hello-robot/stretch_ai#language-directed-pick-and-place).

## Develop with the Docker Installation

You can use the same Docker installation for development on your GPU computer. To do so, you can mount your local copy of the *stretch-ai* repository on your GPU computer into the *stretch-ai* container running on your GPU computer. This enables you to edit the files on your local copy of the *stretch-ai* repository and run them in the container.

Providing the `--dev` command line argument to the GPU computer Docker startup script runs the Docker container with your local copy of the *stretch-ai* repository `../stretch_ai` mounted to the `/app` container directory. To use the script, run the following command on your desktop or laptop:

```bash
./scripts/run_docker_gpu_machine.sh --dev
```

Once the container has started, you should run the following command in the GPU computer's terminal to perform an [editable installation with pip](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) with your local version of the *stretch-ai* code.

```
pip install -e src
```

Now, the apps will run from your local directory. You can edit the apps in your local directory and run them to see the result.

**Please be aware that changes to the Docker container's files and the Docker container's state will be lost once you stop the container. Only changes to the mounted local directory will persist.**
