#!/bin/bash

set -e

# Initialize variables
yes_flag=false
conda_env=""

# Parse command-line options
while getopts ":y" opt; do
    case ${opt} in
        y )
            yes_flag=true
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            exit 1
            ;;
    esac
done

# Check if running in a conda/mamba environment
if [[ -n $CONDA_DEFAULT_ENV ]]; then
    conda_env=$CONDA_DEFAULT_ENV
elif [[ -n $MAMBA_ROOT_PREFIX ]]; then
    conda_env=$(basename "$MAMBA_ROOT_PREFIX")
else
    echo "Error: No conda/mamba environment detected. Please activate an environment before running this script."
    exit 1
fi

echo "Using conda/mamba environment: $conda_env"

# Function to ask for confirmation
confirm() {
    if [ "$yes_flag" = true ]; then
        return 0
    fi
    read -r -p "${1:-Are you sure?} [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            true
            ;;
        *)
            false
            ;;
    esac
}

if confirm "Do you want to install Robosuite and Robocasa?"; then
    # Install Robosuite
    git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
    cd robosuite
    pip install -e .
    cd ..

    # Install Robocasa
    git clone https://github.com/robocasa/robocasa
    cd robocasa
    pip install -e .
    cd ..
else
    echo "Skipping Robosuite and Robocasa installation"
    # Quit
    exit 1
fi

# Download assets
python robocasa/scripts/download_kitchen_assets.py

# Setup macros
if confirm "Do you want to set up system variables and macros for Robocasa?"; then
    python robocasa/scripts/setup_macros.py
fi

# Clone, cd, and install stretch_mujoco repository
if git clone git@github.com:hello-robot/stretch_mujoco.git; then
    echo "Successfully cloned stretch_mujoco repository"
    cd stretch_mujoco
    if pip install -e .; then
        echo "Successfully installed stretch_mujoco"
    else
        echo "Failed to install stretch_mujoco"
        exit 1
    fi
    cd ..
else
    echo "Failed to clone stretch_mujoco repository"
    exit 1
fi

echo "Installation complete!"
