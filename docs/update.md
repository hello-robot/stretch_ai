## Updating your robot

This is a set of steps used to make sure your robot hardware is up to date. We will assume you have installed:

- [stretch_ai](https://github.com/hello-robot/stretch_ai/)
- [stretch_urdf](https://github.com/hello-robot/stretch_urdf/)

Code installed from git must be updated manually as below.

### Step 1: Update the stretch_ai repository

1. Open a terminal and navigate to the `stretch_ai` directory.
2. Run the following commands to update the repository:

```bash
git pull
git submodule update --init --recursive
```

You can install with pip:

```bash
python -m pip install -e .[dev]
```

### Step 2: Update pip packages

1. Open a terminal and run the following command to update all pip packages:

```bash
pip install --upgrade hello-robot-stretch-body hello-robot-stretch-urdf
```

### Step 2: Update the stretch_urdf repository

1. Open a terminal and navigate to the `stretch_urdf` directory.
2. Run the following commands to update the repository:

```bash
git pull
```

Then run the update tool:

```
stretch_urdf_ros_update.py
stretch_urdf_ros_update.py --ros2_rebuild
```
