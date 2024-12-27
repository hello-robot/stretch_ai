# Running Stretch AI on the NVIDIA Jetson

These are experimental instructions for running on the NVIDIA Jetson. These will guide you through how we can build a docker image and use it to run Stretch AI on the Jetson.

In particular, they were tested on the NVIDIA Jetson Orin AGX dev kit.

## About Jetson

Jetson is an embedded device for use on robots, build on the NVIDIA Tegra architecture. It's a powerful device that can run AI models on the edge, and is a popular choice for robotics applications.

However, using the Jetson comes with some trade-offs. In particular, it's not built opn the common `x86` architecture, so some software may not be compatible. Additionally, the Jetson has limited resources, so it may not be able to run all models.

Instead, you generally use Jetson with a custom version of Ubuntu, called `Jetpack`. This is a stripped-down version of Ubuntu that is optimized for the Jetson hardware.

There are two options:
  - Download a docker container with Stretch AI [pre-installed](#running-stretch-ai-in-a-docker-container) (easy!)
  - Install code on the jetson itself - [instructions](#installing-stretch-ai-on-the-jetson) (harder; work in progress)

## Running Stretch AI in a Docker Container

We provide a Stretch AI docker image for the Jetson Orin, which you can use to run Stretch AI on your Jetson.

*Why a docker image?* Many things - especially anything using CUDA - need to be rebuilt specifically for use on the Jetson, both to use Jetpack and to use the Arm-based Tegra architecture. This can be a pain to do manually, so we provide a docker image that has everything pre-built.

You can also download the image [from Docker Hub](https://hub.docker.com/repository/docker/hellorobotinc/stretch-ai_jetson/general) and use it directly instead of building it. You can do this with:

```
./scripts/run_stretch_ai_jetson.sh
```

You can [look at the script](https://github.com/hello-robot/stretch_ai/blob/devel/scripts/run_stretch_ai_jetson.sh) to see how it works.

When you run it, you should see something like this:
```
cpaxton@caliban:~/src/stretch_ai$ ./scripts/run_stretch_ai_jetson.sh 
====================================================
Running Stretch AI docker container with GPU support
$DISPLAY was not set. It has been set to :0 -- please verify that this is correct or GUI will not work!
Reading version from /home/cpaxton/src/stretch_ai/src/stretch/version.py
Source version: 0.2.6
Docker image version: latest
Running docker image hellorobotinc/stretch-ai_cuda-11.8:latest
Running in non-dev mode, not mounting any directory
====================================================
Running docker container with GPU support
 - mounting data at /home/cpaxton/data
User is in Docker group. Running command without sudo.
root@caliban:/stretch_ai#
```

You can then run Stretch AI commands from the container. For example, you can run the `llm_agent` with:

```
root@caliban:/stretch_ai# python3 -m stretch.app.llm_agent
```

Make sure to use the `python3` command instead of `python`.

## Building the Docker Image

If you want to build the docker image yourself, you can do so with the following command:

```
./docker/build-jetson-docker.sh
```

The Dockerfile is located [here](docker/Dockerfile.jetson), in the `docker` directory.

Of particular note is this section at the beginning:
```Dockerfile
FROM dustynv/l4t-text-generation:r35.3.1
```

You can check your Jetpack version directly with:
```bash
cpaxton@caliban:~/src/stretch_ai$ cat /etc/nv_tegra_release
# R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t186ref, EABI: aarch64, DATE: Tue Aug  1 19:57:35 UTC 2023
```

If it does not match, you might need to rebuild the docker image with the correct base image.

## Installing Stretch AI on the Jetson

If you want to manually install things, the process is more difficult. We do not recommend this unless you are comfortable with the process. These instructions are a work in progress and subject to change; you may need to adapt them to your specific setup.

### Install System dependences

Torch, torchvision, and torchaudio have multiple `apt` dependencies that need to be installed first.
```
sudo apt-get install -y  python3-pip libopenblas-dev

# Torchaudio
sudo apt install -y ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev

# Python pip dependences
# These are necessary to build libraries like Torch Audio
python -m pip install --upgrade cmake ninja
```

### Install PyTorch

For NVIDIA tegra chips, you will need to use specific, pre-built wheels for certain versions of Pytorch.

You should check the [official NVIDIA pytorch install instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#prereqs-install).

#### Make sure pytorch installation worked

Cuda should be available and you can put things on the device. Type `python` and try:

```
>>> import torch
>>> torch.cuda.is_available()
True
>>> r = torch.rand(3, 3).cuda()
>>> r
tensor([[0.4269, 0.2103, 0.8359],
        [0.4264, 0.1238, 0.7855],
        [0.5410, 0.6966, 0.2426]], device='cuda:0')
>>> r * r
tensor([[0.1822, 0.0442, 0.6987],
        [0.1818, 0.0153, 0.6170],
        [0.2927, 0.4852, 0.0589]], device='cuda:0')
```

### Optional: install `jtop`

NVIDIA Tegra devices do not support `nvidia-smi`, so you probably want to install `jtop`:

```
sudo pip3 install -U jetson-stats
```

The `jtop` tool requires superuser permissions; don't forget the `-U` flag to make sure you don't need superuser permissions at runtime.

### Install Detic

Detic is an object-detection library that works fairly fast, and that we've had good luck with in [Stretch AI](https://github.com/hello-robot/stretch_ai/) projects. You can see it in the [llm_agent](llm_agent) docs and code.

For this, it may be useful to install timm from [our timm fork](https://github.com/cpaxton/pytorch-image-models/tree/cpaxton/timm-no-torch) which removes the `pytorch` dependency, so that you don't accidentally override your "good" version of the pytorch library.

```
git clone https://github.com/cpaxton/pytorch-image-models.git --branch cpaxton/timm-no-torch
cd pytorch-image-models
python -m pip install -e .
```

### Install Stretch AI

To install Stretch AI from source, add it to the `PYTHONPATH`:

```
git clone git@github.com:hello-robot/stretch_ai.git
export PYTHONPATH=$PYTHONPATH:/path/to/stretch_ai
```

This is to make sure that you don't accidentally upgrade one of the very specific versions of the libraries that you need!

After each step, make sure CUDA still works, and that you have the right versions of the libraries installed.

Your torch version will look something like this:
```
cpaxton@caliban:~/src/stretch_ai$ pip show torch
Name: torch
Version: 2.0.0.nv23.05
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /usr/local/lib/python3.8/dist-packages
Requires: filelock, jinja2, networkx, sympy, typing-extensions
Required-by: torchaudio, torchvision, virgil
```
