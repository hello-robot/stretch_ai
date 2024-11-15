# Docker

This is a guide to setting up a Docker container for running the Stretch software. This is useful for running the software on a computer that doesn't have the correct dependencies installed, or for running the software in a controlled environment.

Hello Robot has a set of scripts for [building robot-side docker images](https://github.com/hello-robot/stretch_docker), which can be conconsidered separately. Here, we will go through what docker is, why we use it, and how to set it up on your client side to run AI code.

## What is Docker and Why Would I Use It?

Docker is a tool that allows you to run software in a container. A container is a lightweight, standalone, executable package of software that includes everything needed to run the software, including the code, runtime, system tools, libraries, and settings. Containers are isolated from each other and from the host system, so they are a great way to run software in a controlled environment.

In particular, this docker image is designed to use the correct CUDA version for the Stretch software. This is useful if you are running the software on a computer that doesn't have the correct CUDA version installed, and it makes installing the right versions of AI/ML libraries much easier.

## Installing Docker

To install Docker, follow the instructions on the [Docker website](https://docs.docker.com/get-docker/). Test by running:

```
docker run hello-world
```

### Troubleshooting

If you are having trouble installing Docker, check the [Docker documentation](https://docs.docker.com/get-docker/) for troubleshooting tips. If you are getting a permission error on Ubuntu, make sure you have added your user to the `docker` group:

```
# Make sure the group exists
sudo groupadd docker

# Add $USER to the docker group
sudo usermod -aG docker $USER

# Restart the Docker daemon so changes take effect
sudo systemctl restart docker
```

If necessary, you can apply new group changes to current session:

```
newgrp docker
```

## Building the Docker Image

To build the Docker image, clone this repository and run the following command in the root directory of the repository:

```bash
docker build -t stretch-ai_cuda-11.8:latest .
```

### Use The Docker Build Script

There is a [docker build script](scripts/build-docker.sh) that can be used to build the Docker image. This script will build the Docker image with the correct CUDA version and tag it with the correct name. To use the script, run:

```bash
./scripts/build-docker.sh
```

from the root directory of the repository.

For more details on the build script, see the [Docker Build Script](scripts/build-docker.md) documentation. You can also continue on into the next section.

### Building and Pushing to Dockerhub

This will use the Hello Robot account as an example (username: `hellorobotinc`). Login with:

```
docker login -u hellorobotinc
```

and enter a password (or create an [access token](https://hub.docker.com/settings/security)).

Then, build the image with:

```bash
docker build -t hellorobotinc/stretch-ai_cuda-11.8:latest .
docker push hellorobotinc/stretch-ai_cuda-11.8:latest
```

You can pull with:

```bash
docker pull hellorobotinc/stretch-ai_cuda-11.8:latest
```

## Running the Docker Image

To run the docker image, we need to:

1. Run a container and attach to the shell
1. Initialize conda and exit the container
1. Start the container again and reconnect to the container shell
1. Activate the conda environment

### 1. Run a container and attach to the shell

The network=host argument makes the container to use your LAN, so it can see your robot
And to have GUI visualization through X server grant root permission to `xhost` and provide `DISPLAY` environment.

```bash
xhost si:localuser:root
docker run \
    -it \
    --gpus all \
    --network host \
    --env DISPLAY "$DISPLAY" \
    hellorobotinc/stretch-ai_cuda-11.8:latest
```

### 2. Initialize conda and exit the container

```bash
conda init # inside the container
exit
```

### 3. Start the container again and reconnect to the container shell

```bash
docker ps -a # get container ID or name of the container just launched, but is now exited
docker start <container-id> # or <container-name>
docker attach <container-id>
```

### 4. Activate the conda environment

```bash
conda activate stretch_ai
```

### 5. Verify container functionality

```bash
# Torch can use GPU
python3
import torch
torch.cuda.is_available() # should return True

# Run view-images demo (make sure server is running on robot)
python3 -m stretch.app.view_images --robot_ip $ROBOT_IP
```

### Tips for Windows 11

If you happen to be running on Windows 11 with WSL2, running the container with the following command will allow you to have GUI forwarded properly. ([source](https://stackoverflow.com/questions/73092750/how-to-show-gui-apps-from-docker-desktop-container-on-windows-11))

```bash
docker run -it -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix `
    -v /run/desktop/mnt/host/wslg:/mnt/wslg `
    -e DISPLAY=:0 `
    -e WAYLAND_DISPLAY=wayland-0 `
    -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir `
    -e PULSE_SERVER=/mnt/wslg/PulseServer `
    --gpus all `
    --network host `
    hellorobotinc/stretch-ai_cuda-11.8:latest
```

### Developing within Docker Container Environment

If you want to use the Docker container as a development environment and retain the changes made in the root `stretch_ai` repository, run the Docker container with the following argument to mount the cloned `stretch_ai` repository from your host filesystem to the `/app` directory inside the Docker container.

```bash
docker run -v ~/stretch_ai:/app [other_docker_options]
```

By mounting the repository this way, any changes you make to the files in the `stretch_ai` directory on your host will be immediately reflected in the `/app` directory inside the container. This allows you to see your changes live, run them and ensures they are not lost when you stop the container.
