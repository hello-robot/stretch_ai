# StretchPy

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)


Tested with Python 3.9/3.10/3.11. **Development Notice**: The code in this repo is a work-in-progress. The code in this repo may be unstable, since we are actively conducting development. Since we have performed limited testing, you may encounter unexpected behaviors.

## Quickstart

This package is not yet available on PyPi. Clone this repo on your Stretch and PC, and install it locally using pip:

```
cd stretchpy/src
pip3 install .
```

On your Stretch, start the server:

```
python3 -m stretch.serve
```

On your PC, add the following yaml to `~/.stretch/config.yaml`:

```yaml
robots:
  - ip_addr: 192.168.1.14 # Substitute with your robot's ip address
    port: 20200
```

Then, on your PC, write some code:

```python
import stretch
stretch.connect()

stretch.move_by(joint_arm=0.1)

for img in stretch.stream_nav_camera():
    cv2.imshow('Nav Camera', img)
    cv2.waitKey(1)
```

Check out the docs on:
 - [Getting status](./docs/status.md)

## Advanced Installation

If you want to install AI code using pytorch, run the following on your GPU-enabled workstation:
```
./install.sh
```

Caution, it may take a while! Several libraries are built from source to avoid potential compatibility issues.

You may need to configure some options for the right pytorch/cuda version. Make sure you have CUDA installed on your computer, preferrably 11.8.

Open3D is an optional dependency used by some 3d visualizations. It does not work in Python 3.12, as of April 2024. Install it with:
```
pip install open3d
```

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

## Example Apps

### Dex Teleop + DobbE

On the robot:
```bash
python -m stretch.demos.dex_teleop.send_d405_images -r
python -m stretch.demos.dex_teleop.follower
```

On a remote, GPU-enabled laptop or workstation connected to the [dex telop setup](https://github.com/hello-robot/stretch_dex_teleop):
```bash
python -m stretch.demos.dex_teleop.leader
```

[Read the Dex Teleop documentation](docs/dex_teleop.md) for more details.

### OVMM Exploration

```bash
python -m stretch.demos.ovmm.run
```

You can show visualizations with:
```bash
python -m stretch.demos.ovmm.run --show-intermediate-maps --show-final-map
```
The flag `--show-intermediate-maps` shows the 3d map after each large motion (waypoint reached), and `--show-final-map` shows the final map after exploration is done.

It will record a PCD/PKL file which can be interpreted with the `read_sparse_voxel_map` script; see below.

### Voxel Map Visualization

You can test the voxel code on a captured pickle file:
```bash
python -m stretch.demos.ovmm.read_sparse_voxel_map -i ~/Downloads/stretch\ output\ 2024-03-21/stretch_output_2024-03-21_13-44-19.pkl
```

Optional open3d visualization of the scene:
```bash
python -m stretch.demos.ovmm.read_sparse_voxel_map -i ~/Downloads/stretch\ output\ 2024-03-21/stretch_output_2024-03-21_13-44-19.pkl  --show-svm
```

### 

## Development

Clone this repo on your Stretch and PC, and install it locally using pip with the "editable" flag:

```
cd stretchpy/src
pip install -e .[dev]
pre-commit install
```

Then follow the quickstart section. See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
