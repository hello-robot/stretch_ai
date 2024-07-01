#!/bin/bash
VERSION=0.0.2
echo "Building docker image with tag hellorobotinc/stretch-ai_cuda-11.8:$VERSION"
echo "Is this ok? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Building docker image..."
else
    echo "Exiting..."
    exit 1
fi
docker build -t hellorobotinc/stretch-ai_cuda-11.8:$VERSION .
docker push hellorobotinc/stretch-ai_cuda-11.8:$VERSION
docker tag hellorobotinc/stretch-ai_cuda-11.8:$VERSION hellorobotinc/stretch-ai_cuda-11.8:latest
docker push hellorobotinc/stretch-ai_cuda-11.8:latest
