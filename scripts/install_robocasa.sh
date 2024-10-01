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

# Install Robosuite
if confirm "Do you want to install Robosuite?"; then
    git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
    cd robosuite
    pip install -e .
    cd ..
fi

# Install Robocasa
if confirm "Do you want to install Robocasa?"; then
    git clone https://github.com/robocasa/robocasa
    cd robocasa
    pip install -e .
    cd ..
fi

# Download assets
if confirm "Do you want to download assets (around 5GB)?"; then
    python robocasa/scripts/download_kitchen_assets.py
fi

# Setup macros
if confirm "Do you want to set up system variables?"; then
    python robocasa/scripts/setup_macros.py
fi

echo "Installation complete!"

