#!/usr/bin/env bash
# This script (c) 2024 Chris Paxton under the MIT license: https://opensource.org/licenses/MIT
# This script is designed to install the HomeRobot/StretchPy environment.
export CUDA_VERSION=11.8
export PYTHON_VERSION=3.10
CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION

script_dir="$(dirname "$0")"
VERSION=`python $script_dir/src/stretch/version.py`
CPU_ONLY="false"
NO_REMOVE="false"
NO_SUBMODULES="false"
MAMBA=mamba
NO_VERSION="false"
# Two cases: -y for yes, --cpu for cpu only
# One more: --conda for conda
for arg in "$@"
do
    case $arg in
        -y|--yes)
            yn="y"
            SKIP_ASKING="true"
            shift
            ;;
        --cpu)
            CPU_ONLY="true"
            shift
            ;;
        --conda)
            MAMBA=conda
            shift
            ;;
        --no-remove)
            NO_REMOVE="true"
            shift
            ;;
        --no-submodules)
            NO_SUBMODULES="true"
            shift
            ;;
        --no-version)
            NO_VERSION="true"
            shift
            ;;
        *)
            shift
            # unknown option
            ;;
    esac
done

# If cpu only, set the cuda version to cpu
if [ "$CPU_ONLY" == "true" ]; then
    export CUDA_VERSION=cpu
    export CUDA_VERSION_NODOT=cpu
    export CUDA_HOME=""
    if [ "$NO_VERSION" == "true" ]; then
        ENV_NAME=stretch_ai_cpu
    else
        ENV_NAME=stretch_ai_cpu_${VERSION}
    fi
    ENV_NAME=stretch_ai_cpu_${VERSION}
    export PYTORCH_VERSION=2.1.2
else
    export CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
    if [ "$NO_VERSION" == "true" ]; then
        ENV_NAME=stretch_ai
    else
        ENV_NAME=stretch_ai_${VERSION}
    fi
    export PYTORCH_VERSION=2.3.1
fi

echo "=============================================="
echo "         INSTALLING STRETCH AI TOOLS"
echo "=============================================="
echo "---------------------------------------------"
echo "Environment name: $ENV_NAME"
echo "PyTorch Version: $PYTORCH_VERSION"
echo "CUDA Version: $CUDA_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "CUDA Version No Dot: $CUDA_VERSION_NODOT"
echo "Using tool: $MAMBA"
echo "---------------------------------------------"
echo "Notes:"
echo " - This script will remove the existing environment if it exists."
echo " - This script will install the following packages:"
echo "   - pytorch=$PYTORCH_VERSION"
echo "   - pytorch-cuda=$CUDA_VERSION"
echo "   - torchvision"
if [ $INSTALL_TORCH_GEOMETRIC == "true" ]; then
    echo "   - torch-geometric"
    echo "   - torch-cluster"
    echo "   - torch-scatter"
fi
echo "   - python=$PYTHON_VERSION"
echo " - Python version 3.12 is not supported by Open3d."
echo "---------------------------------------------"
echo "Currently:"
echo " - CUDA_HOME=$CUDA_HOME"
echo " - python=`which python`"


# if -y flag was passed in, do not bother asking
#
if [ "$SKIP_ASKING" == "true" ]; then
    yn="y"
else
    read -p "Does all this look correct? (y/n) " yn
    case $yn in
        y ) echo "Starting installation..." ;;
        n ) echo "Exiting...";
            exit ;;
        * ) echo Invalid response!;
            exit 1 ;;
    esac
fi

# Exit immediately if anything fails
set -e

# Install git-lfs
echo "Installing git-lfs..."
echo "If this fails, install git-lfs with:"
echo ""
echo "     sudo apt-get install git-lfs"
echo ""
git lfs install

# Only remove if NO_REMOVe is false
if [ "$NO_REMOVE" == "false" ]; then
    echo "Removing existing environment..."
    $MAMBA env remove -n $ENV_NAME -y
fi
# If using cpu only, create a separate environment
if [ "$CPU_ONLY" == "true" ]; then
    $MAMBA create -n $ENV_NAME -c pytorch pytorch=$PYTORCH_VERSION torchvision torchaudio cpuonly python=$PYTHON_VERSION -y
else
    # Else, install the cuda version
    $MAMBA create -n $ENV_NAME -c pytorch -c nvidia pytorch=$PYTORCH_VERSION pytorch-cuda=$CUDA_VERSION torchvision torchaudio python=$PYTHON_VERSION -y
fi

source activate $ENV_NAME

echo "Install a version of setuptools for which clip works."
pip install setuptools==69.5.1

echo ""
echo "---------------------------------------------"
echo "---- INSTALLING STRETCH AI DEPENDENCIES  ----"
echo "Will be installed via pip into env: $ENV_NAME"

pip install -e ./src[dev]

echo ""
echo "---------------------------------------------"
echo "----   INSTALLING DETIC FOR PERCEPTION   ----"
# echo "The third_party folder will be removed!"
if [ "$SKIP_ASKING" == "true" ]; then
    echo "Proceeding with installation because you passed in the -y flag."
    yn="y"
else
    read -p "Do you want to proceed? (y/n) " yn
    case $yn in
        y ) echo "Starting installation..." ;;
        n ) echo "Exiting...";
            exit ;;
        * ) echo Invalid response!;
            exit 1 ;;
    esac
fi

# If not cpu only, then we can use perception
# OR if no submodules, then we can't install perception
if [ "$CPU_ONLY" == "true" ] || [ "$NO_SUBMODULES" == "true" ]; then
    echo "Skipping perception installation for CPU only"
else
    echo "Install detectron2 for perception (required by Detic)"
    git submodule update --init --recursive
    cd third_party/detectron2
    pip install -e .

    echo "Install Detic for perception"
    cd ../../src/stretch/perception/detection/detic/Detic
    # Make sure it's up to date
    git submodule update --init --recursive
    pip install -r requirements.txt

    # cd ../../src/stretch/perception/detection/detic/Detic
    # Create folder for checkpoints and download
    mkdir -p models
    echo "Download DETIC checkpoint..."
    wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
fi

echo ""
echo "=============================================="
echo "         INSTALLATION COMPLETE"
echo "Finished setting up the StretchPy environment."
echo "Environment name: $ENV_NAME"
echo "CUDA Version: $CUDA_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "CUDA Version No Dot: $CUDA_VERSION_NODOT"
echo "CUDA_HOME=$CUDA_HOME"
echo "python=`which python`"
echo "You can start using it with:"
echo ""
echo "     source activate $ENV_NAME"
echo "=============================================="
