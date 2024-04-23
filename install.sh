#!/usr/bin/env bash
# This script (c) 2024 Chris Paxton under the MIT license: https://opensource.org/licenses/MIT
# This script is designed to install the HomeRobot/StretchPy environment.
export PYTORCH_VERSION=2.1.2
export CUDA_VERSION=11.8
export PYTHON_VERSION=3.10
ENV_NAME=stretchpy
CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION
echo "=============================================="
echo "         INSTALLING STRETCH AI TOOLS"
echo "=============================================="
echo "---------------------------------------------"
echo "Environment name: $ENV_NAME"
echo "PyTorch Version: $PYTORCH_VERSION"
echo "CUDA Version: $CUDA_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "CUDA Version No Dot: $CUDA_VERSION_NODOT"
echo "---------------------------------------------"
echo "Currently:"
echo " - CUDA_HOME=$CUDA_HOME"
echo " - python=`which python`"
echo "---------------------------------------------"
read -p "Does all this look correct? (y/n) " yn
case $yn in
	y ) echo "Starting installation...";;
	n ) echo "Exiting...";
		exit;;
	* ) echo Invalid response!;
		exit 1;;
esac
mamba env remove -n $ENV_NAME -y
mamba create -n $ENV_NAME -c pyg -c pytorch -c nvidia pytorch=$PYTORCH_VERSION pytorch-cuda=$CUDA_VERSION pyg torchvision python=$PYTHON_VERSION -y 
source activate $ENV_NAME
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install torch_cluster -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION_NODOT}.html
pip install torch_scatter -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION_NODOT}.html
pip install torch_geometric
pip install -e ./src[dev]
# Open3d is an optional dependency - not included in setup.py since not supported on 3.12
pip install open3d
