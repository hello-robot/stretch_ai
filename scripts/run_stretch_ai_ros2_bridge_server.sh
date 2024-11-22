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
no_d405=false

for arg in "$@"
do
    case $arg in
        --update)
            update=true
            shift
            ;;
        --no-d405)
            no_d405=true
            shift
            ;;
    esac
done

echo "Starting Stretch AI ROS2 Bridge Server on $HELLO_FLEET_ID"
echo "========================================================="
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

# echo "Reading version from $parent_dir/src/stretch/version.py"
VERSION=`python3 $parent_dir/src/stretch/version.py`
echo "Source version: $VERSION"

VERSION="latest"
echo "Docker image version: $VERSION"

# sudo chown -R $USER:$USER /home/$USER/stretch_user
# sudo chown -R $USER:$USER /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf

echo "Running docker image hellorobotinc/stretch-ai-ros2-bridge:$VERSION"
# Make sure the image is up to date
# Update the Docker image if --update flag is set
if $update; then
    run_docker_command pull hellorobotinc/stretch-ai-ros2-bridge:$VERSION
fi

if $no_d405; then
    launch_command="ros2 launch stretch_ros2_bridge server_no_d405.launch.py"
else
    launch_command="ros2 launch stretch_ros2_bridge server.launch.py"
fi

# Run the container
# The options mean:
# --net=host: use host network, so that the container can communicate with the robot
# --privileged=true: allow privileged mode, needed for some devices
# -v /dev:/dev: mount the /dev directory
# --device /dev/snd: allow sound devices
# -e DISPLAY="$DISPLAY": set the display environment variable
# -v /tmp/.X11-unix:/tmp/.X11-unix: mount the X11 socket for GUI
# -v /run/dbus/:/run/dbus/:rw: mount the dbus socket for communication
# -v /dev/shm:/dev/shm: mount the shared memory directory
# --group-add=audio: add the audio group to the container
# -v /home/$USER/stretch_user:/home/hello-robot/stretch_user_copy: mount the user directory
# -v /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf:/home/hello-robot/stretch_description/share/stretch_description/urdf: mount the urdf directory
# -e HELLO_FLEET_ID=$HELLO_FLEET_ID: set the fleet ID
# hellorobotinc/stretch-ai-ros2-bridge:$VERSION: the docker image to run
# bash -c "source /home/hello-robot/.bashrc; cp -rf /home/hello-robot/stretch_user_copy/* /home/hello-robot/stretch_user; export HELLO_FLEET_ID=$HELLO_FLEET_ID; ros2 launch stretch_ros2_bridge server_no_d405.launch.py": run the server
run_docker_command run -it --rm \
    --net=host \
    --privileged=true \
    -v /dev:/dev \
    --device /dev/snd \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /run/dbus/:/run/dbus/:rw \
    -v /dev/shm:/dev/shm \
    --group-add=audio \
    -v /home/$USER/stretch_user:/home/hello-robot/stretch_user_copy \
    -v /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf:/home/hello-robot/stretch_description/share/stretch_description/urdf \
    -e HELLO_FLEET_ID=$HELLO_FLEET_ID \
    hellorobotinc/stretch-ai-ros2-bridge:$VERSION \
    bash -c "source /home/hello-robot/.bashrc; cp -rf /home/hello-robot/stretch_user_copy/* /home/hello-robot/stretch_user; export HELLO_FLEET_ID=$HELLO_FLEET_ID; $launch_command"
