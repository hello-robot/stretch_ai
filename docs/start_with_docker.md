# Starting with Docker

Docker is a tool that allows you to package an application and its dependencies in a virtual container that can run on any Linux server. This ensures that the application will always run the same, regardless of the environment it is running in.

Stretch AI uses Docker to package the software and its dependencies in a container. This makes it easy to run the software on any computer that has Docker installed.

## Installing Docker

Start by installing docker on the robot and your desktop or GPU laptop:
```
sudo apt-get update
sudo apt-get install docker.io
```

On the GPU machine, you also need the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Check the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the most up-to-date instructions.

You can also install with:
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Then restart the docker service:
```
sudo systemctl restart docker
```

An [nvidia docker install script](scripts/install_nvidia_container_toolkit.sh) has been provided for Ubuntu machines. Again, though, [check the official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the most up-to-date instructions and if you have any issues.

## On the Robot

We provide an easy startup script for running the Stretch AI software in a Docker container. To use the script, run the following command on the robot:

```
./scripts/run_stretch_ai_ros2_bridge_server.sh
```

You will see something like this as the docker image is downloaded:
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

This may take some time!

## On the Desktop or GPU Laptop

Now, to run the docker image, we need to:

1. Run a container and attach to the shell
1. Initialize conda and exit the container
1. Start the container again and reconnect to the container shell
1. Activate the conda environment


### Run the container for the first time

```bash
./scripts/run_docker_gpu_machine.sh
```

#### Verify NVIDIA docker
Make sure nvidia docker is set up correctly. To do this, run the `nvidia-smi` command in the docker shell.

You should see something like this:
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

#### Use the conda environment

```bash
conda init # inside the container
```

## Testing
