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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

# Get the version of the docker image
echo "Reading version from $parent_dir/src/stretch/version.py"
VERSION=`python $parent_dir/src/stretch/version.py`

echo "Running docker image hellorobotinc/stretch-ai_cuda-11.8:$VERSION"

# Vérifier si le drapeau --dev est présent
if [[ "$*" == *"--dev"* ]]; then
    mount_option="-v $parent_dir:/app"
    echo "Mounting $parent_dir into /app"
else
    mount_option=""
    echo "Running in non-dev mode, not mounting any directory"
fi

echo "===================================================="

sudo docker run \
    -it \
    --gpus all \
    --network host \
    --env DISPLAY="$DISPLAY" \
    $mount_option \
    hellorobotinc/stretch-ai_cuda-11.8:$VERSION

