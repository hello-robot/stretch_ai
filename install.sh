#!/usr/bin/env bash
# This script (c) 2024 Hello Robot under the MIT license: https://opensource.org/licenses/MIT
# This script is designed to install the HomeRobot/StretchPy environment.
export CUDA_VERSION=11.8
export PYTHON_VERSION=3.10

script_dir="$(dirname "$0")"
VERSION=`python3 $script_dir/src/stretch/version.py`
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
        --cuda=*)
            CUDA_VERSION="${arg#*=}"
            echo "Setting CUDA Version: $CUDA_VERSION"
            shift
            ;;
        *)
            shift
            # unknown option
            ;;
    esac
done

CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION

# Check if the user has the required packages
# If not, install them
# If these packages are not installed, you will run into issues with pyaudio
sudo apt-get update
echo "Checking for required packages: "
echo "     libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg"
echo "If these are not installed, you will run into issues with pyaudio."
if [ "$SKIP_ASKING" == "true" ]; then
    sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1 -y
else
    sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 espeak ffmpeg build-essential wget unzip libsndfile1
fi

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
else
    export CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
    if [ "$NO_VERSION" == "true" ]; then
        ENV_NAME=stretch_ai
    else
        ENV_NAME=stretch_ai_${VERSION}
    fi
fi

echo "=============================================="
echo "         INSTALLING STRETCH AI TOOLS"
echo "=============================================="
echo "---------------------------------------------"
echo "Environment name: $ENV_NAME"
# echo "PyTorch Version: $PYTORCH_VERSION"
echo "CUDA Version: $CUDA_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "CUDA Version No Dot: $CUDA_VERSION_NODOT"
echo "Using tool: $MAMBA"
echo "---------------------------------------------"
echo "Notes:"
echo " - This script will remove the existing environment if it exists."
echo " - This script will install the following packages:"
# echo "   - pytorch=$PYTORCH_VERSION"
echo "   - pytorch-cuda=$CUDA_VERSION"
echo "   - torchvision"
if [[ $INSTALL_TORCH_GEOMETRIC == "true" ]]; then
    echo "   - torch-geometric"
    echo "   - torch-cluster"
    echo "   - torch-scatter"
fi
echo "   - python=$PYTHON_VERSION"
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
    $MAMBA env remove -n $ENV_NAME -y || true
fi

$MAMBA create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activate env $ENV_NAME"

# If you don't install conda and only have mamba, please don't use this script from here and manually install everything.
eval "$(conda shell.bash hook)"
source activate $ENV_NAME

echo "Install a version of setuptools for which clip works."
python -m pip install setuptools==69.5.1

echo ""
echo "---------------------------------------------"
echo "---- INSTALLING STRETCH AI DEPENDENCIES  ----"
echo "Will be installed via pip into env: $ENV_NAME"

python -m pip install -e ./src[dev] --no-cache-dir

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
echo "     $MAMBA activate $ENV_NAME"
echo "=============================================="
