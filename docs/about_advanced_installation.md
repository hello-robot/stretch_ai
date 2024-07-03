# Advanced Installation

The following instructions are for installing the AI code on your GPU-enabled workstation. This is not necessary for running the robot, but is required for running AI models on your PC.

First, [make sure you have CUDA 11.8 installed](#installing-cuda-11.8). We install a specific version of CUDA to make turnkey installation easier, and because various dependencies are compiled against specific CUDA versions. Then you can run the following on your GPU-enabled workstation:

```bash
./install.sh
```

This will install the necessary dependencies for running AI models on your PC.

## Debugging Common Issues

First, verify that your installation is successful. One of the most common issues with the advanced installation is a CUDA version conflict, which means that Torch cannot run on your GPU.

### Verifying Advanced Installation

The most common issue is with `torch_cluster`, or that cuda is set up wrong. Make sure it runs by starting `python` and running:

```python
import torch_cluster
import torch
torch.cuda.is_available()
torch.rand(3, 3).to("cuda")
```

You should see:

- `torch_cluster` imports successfully
- `True` for `torch.cuda.is_available()`
- No errors for `torch.rand(3, 3).to("cuda")`

If instead you get an error, run the following to check your CUDA version:

```bash
nvcc --version
```

Make sure you have CUDA installed on your computer, preferably 11.8. It's possible to install multiple versions of CUDA on your computer, so make sure you have the correct version installed. You do not need to and should not install new versions of your NVIDIA drivers, but you may want to [install CUDA 11.8](https://developer.nvidia.com/cuda-11.8-download-archive) if you don't have it already, following the instructions in [Installing CUDA 11.8](#installing-cuda-11.8).

## Installing CUDA 11.8

Download the runfile version of CUDA 11.8. E.g. for Ubuntu 22.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

![CUDA Installation](images/cuda_install.png)

Follow the prompts to install CUDA 11.8. When you get to the prompt to install the NVIDIA driver, select "No" to avoid installing a new driver. Also make sure you deselect the prompt for setting the system CUDA version!

Before running the install script, set the `$CUDA_HOME` environment variable to point to the new CUDA installation. For example, on Ubuntu 22.04:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
./install.sh
```

THis should help avoid issues.
