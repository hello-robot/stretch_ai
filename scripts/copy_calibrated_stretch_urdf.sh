#!/bin/bash

# Default user
USER="hello-robot"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot_ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        --user)
            USER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If ROBOT_IP is not set, try to load it from the file
if [ -z "$ROBOT_IP" ]; then
    if [ -f ~/.stretch/robot_ip.txt ]; then
        ROBOT_IP=$(cat ~/.stretch/robot_ip.txt)
    else
        echo "Error: Robot IP not provided and ~/.stretch/robot_ip.txt does not exist."
        exit 1
    fi
fi

# Check if ROBOT_IP is set
if [ -z "$ROBOT_IP" ]; then
    echo "Error: Robot IP is not set."
    exit 1
fi

# Run the scp command
scp "$USER@$ROBOT_IP:~/ament_ws/src/stretch_ros2/stretch_description/urdf/stretch.urdf" $HOME/.stretch/stretch.urdf

# Check if the scp command was successful
if [ $? -eq 0 ]; then
    echo "File successfully copied."
else
    echo "Error: Failed to copy the file."
    exit 1
fi
