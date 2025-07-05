#!/bin/bash
# This script builds the docker image for the Stretch AI ROS2 bridge.
#
# Exit on error.
set -e

# Get the current version of the package.
VERSION=`python src/stretch/version.py`
echo "Building docker image with tag hellorobotinc/stretch-ai_cuda-11.8:$VERSION"

SKIP_ASKING="false"
for arg in "$@"
do
    case $arg in
        -y|--yes)
            yn="y"
            SKIP_ASKING="true"
            shift
            ;;
        *)
            shift
            # unknown option
            ;;
    esac
done
if [ "$SKIP_ASKING" == "false" ]; then
    read -p "Verify that this is correct. Proceed? (y/n) " yn
    if [ "$answer" == "${answer#[Yy]}" ] ;then
        echo "Building docker image..."
    else
        echo "Exiting..."
        exit 1
    fi
fi
# Build the docker image with the current tag.
docker build -t hellorobotinc/stretch-ai_cuda-11.8:$VERSION . -f docker/Dockerfile.cuda-11.8 --no-cache
docker push hellorobotinc/stretch-ai_cuda-11.8:$VERSION
docker tag hellorobotinc/stretch-ai_cuda-11.8:$VERSION hellorobotinc/stretch-ai_cuda-11.8:latest
docker push hellorobotinc/stretch-ai_cuda-11.8:latest
