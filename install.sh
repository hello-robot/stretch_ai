#!/usr/bin/env bash
# This script (c) 2024 Chris Paxton under the MIT license: https://opensource.org/licenses/MIT
# This script is designed to install the HomeRobot/StretchPy environment.
export PYTORCH_VERSION=2.1.2
export CUDA_VERSION=11.8
export PYTHON_VERSION=3.10
CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION

CPU_ONLY="false"
INSTALL_PYTORCH3D="false"
MAMBA=mamba
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
        --pytorch3d)
            INSTALL_PYTORCH3D="true"
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
    ENV_NAME=stretch_ai_cpu
else
    export CUDA_VERSION_NODOT="${CUDA_VERSION//./}"
    ENV_NAME=stretch_ai
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
echo "   - pyg"
echo "   - torchvision"
echo "   - python=$PYTHON_VERSION"
echo " - This script will install the following packages from source:"
echo "   - pytorch3d"
echo "   - torch_scatter"
echo "   - torch_cluster"
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

$MAMBA env remove -n $ENV_NAME -y
# If using cpu only, create a separate environment
if [ "$CPU_ONLY" == "true" ]; then
    $MAMBA create -n $ENV_NAME -c pyg -c pytorch pytorch=$PYTORCH_VERSION pyg torchvision cpuonly python=$PYTHON_VERSION -y
else
    # Else, install the cuda version
    $MAMBA create -n $ENV_NAME -c pyg -c pytorch -c nvidia pytorch=$PYTORCH_VERSION pytorch-cuda=$CUDA_VERSION pyg torchvision python=$PYTHON_VERSION -y
fi

source activate $ENV_NAME

# Now install pytorch3d a bit faster
$MAMBA install -c fvcore -c iopath -c conda-forge fvcore iopath -y

echo "Install a version of setuptools for which pytorch3d and clip work."
pip install setuptools==69.5.1

echo ""
echo "---------------------------------------------"
echo "---- INSTALLING STRETCH AI DEPENDENCIES  ----"
echo "Will be installed via pip into env: $ENV_NAME"

# If not using cpu only, install the following
pip install torch_cluster -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION_NODOT}.html
pip install torch_scatter -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION_NODOT}.html
pip install torch_geometric

# This is no longer necessary but might be useful for some checks
if [ "$INSTALL_PYTORCH3D" == "true" ]; then
    echo "Installing pytorch3d from source"
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

pip install -e ./src[dev]

echo ""
echo "---------------------------------------------"
echo "----   INSTALLING DETIC FOR PERCEPTION   ----"
echo "Will be installed to: $PWD/third_party/Detic"
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
if [ "$CPU_ONLY" == "true" ]; then
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
