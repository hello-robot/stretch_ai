#!/bin/bash

# Description: Run the docker container with GPU support for Stretch AI

# Example usage:
# ./run_stretch_ai_docker.sh                     # Run GPU client
# ./run_stretch_ai_docker.sh --update            # Run GPU client with image update
# ./run_stretch_ai_docker.sh --dev               # Run GPU client in dev mode
# ./run_stretch_ai_docker.sh --ros2-bridge       # Run ROS2 bridge server
# ./run_stretch_ai_docker.sh --ros2-bridge --no-d405  # Run ROS2 bridge server without D405 camera

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

# Get script directory and parent directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

# Get the version of the docker image
echo "Reading version from $parent_dir/src/stretch/version.py"
VERSION=$(python3 $parent_dir/src/stretch/version.py)
echo "Source version: $VERSION"
VERSION="latest"
echo "Docker image version: $VERSION"

# Parse command-line arguments
update=false
dev_mode=false
no_d405=false
ros2_bridge=false

for arg in "$@"
do
    case $arg in
        --update)
            update=true
            shift
            ;;
        --dev)
            dev_mode=true
            shift
            ;;
        --no-d405)
            no_d405=true
            shift
            ;;
        --ros2-bridge)
            ros2_bridge=true
            shift
            ;;
    esac
done

# Check for incompatible flag combinations
if $ros2_bridge && $dev_mode; then
    echo "Error: --ros2-bridge and --dev flags cannot be used together."
    exit 1
fi

if $no_d405 && ! $ros2_bridge; then
    echo "Error: --no-d405 flag can only be used with --ros2-bridge."
    exit 1
fi

# Set the appropriate Docker image based on the mode
if $ros2_bridge; then
    docker_image="hellorobotinc/stretch-ai-ros2-bridge:$VERSION"
    echo "Running docker image $docker_image"
else
    docker_image="hellorobotinc/stretch-ai_cuda-11.8:$VERSION"
    echo "Running docker image $docker_image"
fi

# Update the Docker image if --update flag is set
if $update; then
    echo "Updating Docker image..."
    run_docker_command pull $docker_image
fi

# Set up common Docker run options
docker_run_options="-it --rm --net=host --privileged=true -v /dev:/dev --device /dev/snd -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /run/dbus/:/run/dbus/:rw -v /dev/shm:/dev/shm --group-add=audio"

# Add mode-specific options
if $ros2_bridge; then
    docker_run_options+=" -v /home/$USER/stretch_user:/home/hello-robot/stretch_user_copy"
    docker_run_options+=" -v /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf:/home/hello-robot/stretch_description/share/stretch_description/urdf"
    docker_run_options+=" -e HELLO_FLEET_ID=$HELLO_FLEET_ID"

    if $no_d405; then
        launch_command="ros2 launch stretch_ros2_bridge server_no_d405.launch.py"
    else
        launch_command="ros2 launch stretch_ros2_bridge server.launch.py"
    fi

    docker_command="bash -c \"source /home/hello-robot/.bashrc; cp -rf /home/hello-robot/stretch_user_copy/* /home/hello-robot/stretch_user; export HELLO_FLEET_ID=$HELLO_FLEET_ID; $launch_command\""
else
    if $dev_mode; then
        docker_run_options+=" -v $parent_dir:/app"
        echo "Mounting $parent_dir into /app"
    fi
    docker_run_options+=" --gpus all"
    docker_command=""
fi

# Run the Docker container
echo "Running Docker container..."
run_docker_command run $docker_run_options $docker_image $docker_command

