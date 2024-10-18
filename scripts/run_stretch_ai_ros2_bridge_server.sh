#!/bin/bash
echo "Starting Stretch AI ROS2 Bridge Server on $HELLO_FLEET_ID"
echo "========================================================="
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
VERSION=`python $parent_dir/src/stretch/version.py`
echo "Running docker image hellorobotinc/stretch-ai-ros2-bridge:$VERSION"
sudo docker run -it \
    --net=host \
    --privileged \
    -v /dev:/dev \
    -v /home/hello-robot/stretch_user:/home/hello-robot/stretch_user \
    -v /home/hello-robot/ament_ws/install/stretch_description/share/stretch_description/urdf:/home/hello-robot/stretch_description/share/stretch_description/urdf \
    -e HELLO_FLEET_ID=$HELLO_FLEET_ID \
    hellorobotinc/stretch-ai-ros2-bridge:$VERSION \
    bash -c "source /home/hello-robot/.bashrc; export HELLO_FLEET_ID=$HELLO_FLEET_ID; ros2 launch stretch_ros2_bridge server.launch.py"
