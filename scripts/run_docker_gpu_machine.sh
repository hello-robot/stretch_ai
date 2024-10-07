#!/bin/bash
# Description: Run the docker container with GPU support

echo "===================================================="
echo "Running Stretch AI docker container with GPU support"
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "\$DISPLAY was not set. It has been set to :0 -- please verify that this is correct or GUI will not work!"
else
    echo "\$DISPLAY is already set to $DISPLAY"
fi

xhost si:localuser:root
echo "===================================================="

sudo docker run \
    -it \
    --gpus all \
    --network host \
    --env DISPLAY="$DISPLAY" \
    hellorobotinc/stretch-ai_cuda-11.8:0.0.13
