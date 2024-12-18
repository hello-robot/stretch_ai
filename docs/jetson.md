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



## Installing Stretch AI on the Jetson

Work in progress notes that might be useful:

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


