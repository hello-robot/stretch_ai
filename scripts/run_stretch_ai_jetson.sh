#!/bin/bash
# Description: Run the docker container with GPU support

# Make sure it fails if we see any errors
set -e

# Function to check if user is in Docker group
is_in_docker_group() {
    groups | grep -q docker
}

# Function to run Docker command
run_docker_command() {
    if is_in_docker_group; then
        echo "User is in Docker group. Running command without sudo."
        docker "$@"
    else
        echo "User is not in Docker group. Running command with sudo."
        echo "To run without sudo, add your user to the docker group: sudo usermod -aG docker $USER"
        echo "Then log out and log back in."
        echo "Alternately, you can change for the current shell with newgrp: newgrp docker"
        sudo docker "$@"
    fi
}

# Parse command-line arguments
update=false

for arg in "$@"
do
    case $arg in
        --update)
            update=true
            shift
            ;;
    esac
done

echo "===================================================="
echo "Running Stretch AI docker container with GPU support"
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "\$DISPLAY was not set. It has been set to :0 -- please verify that this is correct or GUI will not work!"
else
    echo "\$DISPLAY is already set to $DISPLAY"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

# Get the version of the docker image
echo "Reading version from $parent_dir/src/stretch/version.py"
VERSION=`python3 $parent_dir/src/stretch/version.py`
echo "Source version: $VERSION"

VERSION="latest"
echo "Docker image version: $VERSION"

echo "Running docker image hellorobotinc/stretch-ai_jetson:$VERSION"

# Check dev flag
if [[ "$*" == *"--dev"* ]]; then
    mount_option="-v $parent_dir:/stretch_ai"
    echo "Mounting $parent_dir into /stretch_ai"
else
    mount_option=""
    echo "Running in non-dev mode, not mounting any directory"
fi

echo "===================================================="
echo "Running docker container with GPU support"
export DATA_DIR=$HOME/data
echo " - mounting data at $DATA_DIR"

# Make sure the image is up to date
# Update the Docker image if --update flag is set
if $update; then
    run_docker_command pull hellorobotinc/stretch-ai_jetson:$VERSION
fi

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
run_docker_command run \
    -it \
    --runtime nvidia \
    --gpus all \
    --shm-size=8g \
    -v /dev:/dev \
    -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
    --device /dev/snd \
    --device /dev/bus/usb \
    --privileged=true \
    --network host \
    --env DISPLAY="$DISPLAY" \
    --env HF_HOME=$HF_HOME \
    --volume /tmp/argus_socket:/tmp/argus_socket \
    --volume /etc/enctune.conf:/etc/enctune.conf \
    --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
    --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
    --volume /var/run/dbus:/var/run/dbus \
    --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume $DATA_DIR:/data \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /run/dbus/:/run/dbus/:rw \
    -v /dev/shm:/dev/shm \
    --group-add=audio \
    $mount_option \
    hellorobotinc/stretch-ai_jetson:$VERSION
