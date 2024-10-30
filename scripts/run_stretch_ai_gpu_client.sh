#!/bin/bash
# Description: Run the docker container with GPU support

# Make sure it fails if we see any errors
set -e

echo "===================================================="
echo "Running Stretch AI docker container with GPU support"
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "\$DISPLAY was not set. It has been set to :0 -- please verify that this is correct or GUI will not work!"
else
    echo "\$DISPLAY is already set to $DISPLAY"
fi

xhost si:localuser:root

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

# Get the version of the docker image
echo "Reading version from $parent_dir/src/stretch/version.py"
VERSION=`python3 $parent_dir/src/stretch/version.py`
echo "Source version: $VERSION"

VERSION="latest"
echo "Docker image version: $VERSION"

echo "Running docker image hellorobotinc/stretch-ai_cuda-11.8:$VERSION"

# Check dev flag
if [[ "$*" == *"--dev"* ]]; then
    mount_option="-v $parent_dir:/app"
    echo "Mounting $parent_dir into /app"
else
    mount_option=""
    echo "Running in non-dev mode, not mounting any directory"
fi

echo "===================================================="
echo "Running docker container with GPU support"
# Make sure the image is up to date
sudo docker pull hellorobotinc/stretch-ai_cuda-11.8:$VERSION
# Run the container
# The options mean:
# -it: interactive mode
# --gpus all: use all GPUs
# -v /dev:/dev: mount the /dev directory
# --device /dev/snd: allow sound devices
# --privileged=true: allow privileged mode, needed for some devices
# --network host: use host network, so that the container can communicate with the robot
# --env DISPLAY="$DISPLAY": set the display environment variable
# -v /tmp/.X11-unix:/tmp/.X11-unix: mount the X11 socket for GUI
# -v /run/dbus/:/run/dbus/:rw: mount the dbus socket for communication
# -v /dev/shm:/dev/shm: mount the shared memory directory
# --group-add=audio: add the audio group to the container
# $mount_option: mount the parent directory if in dev mode
# hellorobotinc/stretch-ai_cuda-11.8:$VERSION: the docker image to run
sudo docker run \
    -it \
    --gpus all \
    -v /dev:/dev \
    --device /dev/snd \
    --privileged=true \
    --network host \
    --env DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /run/dbus/:/run/dbus/:rw \
    -v /dev/shm:/dev/shm \
    --group-add=audio \
    $mount_option \
    hellorobotinc/stretch-ai_cuda-11.8:$VERSION

