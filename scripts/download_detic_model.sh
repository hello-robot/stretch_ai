#!/usr/bin/env bash
# Make sure it fails if we see any errors
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

echo "Install Detic for perception"
cd $parent_dir/src/stretch/perception/detection/detic/Detic
# Make sure it's up to date
git submodule update --init --recursive
# pip install -r requirements.txt

# Create folder for checkpoints and download
mkdir -p models
echo "Download DETIC checkpoint..."
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
