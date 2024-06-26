# StretchPy with HomeRobot

Some demos use [HomeRobot](https://github.com/cpaxton/home-robot) for low-level control on the robot hardware, and use more AI features on whatever offboard compute you have available - for example a laptop with a GPU.

## Installation

To install HomeRobot, run the following on your Stretch. We will roughly follow the [HomeRobot hardware install instructions](https://github.com/facebookresearch/home-robot/blob/main/docs/install_robot.md).

```bash
cd $HOME/src  # or path to wherever you keep source

# Clone the HomeRobot fork with ROS2 support
git clone git@github.com:cpaxton/home-robot.git --branch cpaxton/ros2-migration

# Find the root of the cloned code.
# You will need to add this to your ~/.bashrc as well
HOME_ROBOT_ROOT=$(realpath home-robot)

# Install requirements
cd $HOME_ROBOT_ROOT/src/home_robot
pip install -r requirements.txt
pip install -e .

# Combine it with ROS workspace
# Symlink it into the ament workspace from wherever it is
ln -s $HOME_ROBOT_ROOT/src/robot_hw_python $HOME/ament_ws/src/robot_hw_python

# And now build the package
cd $HOME/ament_ws
colcon build --symlink-install --packages-select=robot_hw_python
source $HOME/ament/install/setup.bash
```

### On the desktop

Clone stretchpy and run the install script:

```
./install.sh
```

Make sure everything is set up properly and it works well. You may need to install CUDA 11.8 for these scripts to work properly. Do not overwrite your drivers when doing so; check the Readme for notes.

## Running the code

### On Hardware

#### Easy Mode

Everything should just work! Run the launch file with

```
ros2 launch robot_hw_python server.launch.py
```

#### Developer Mode

Launch the ROS2 code and the ZMQ server separate windows.

```
ros2 launch robot_hw_python startup_stretch_hector_slam.launch.py
```

In a separate terminal:

```
cd $HOME_ROBOT_ROOT
python src/robot_hw_python/robot_hw_python/remote/server.py
```

### On the Desktop

```
conda activate stretchpy
python -m stretch.app.pickup.run
```

Often it's useful to add the `--reset` flag, which will make the robot BLINDLY move back to (0, 0, 0) - the location from which you started the ros2 launch files above -- before running an experiment,.

```
conda activate stretchpy
python -m stretch.app.pickup.run --reset
```

#### Before you start

- Run the launch files with the robot in a safe location -- one that's got about half a meter free on each side. The motion planning assumes that the area around there the robot starts is safe so that we don't need to tilt the head.
- When you start testing, set up two objects directly in front of the camera: a box and a toy animal of some kind. This means it won't explore and will just go directly to the objects. If the robot starts to wander off, something was not properly detected!
- The robot should explore until it finds a box and a toy animal. This isn't as thoroughly tested though.

## Notes on behavior

### Low level controller

Low level controller is at

```
$HOME_ROBOT_ROOT/src/home-robot/home-robot/control/goto_controller.py
```

Parameters are at: [`src/home_robot/config/control/noplan_velocity_hw.yaml`](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/config/control/noplan_velocity_hw.yaml) and you can modify them there to make sure the robot behaves well.
