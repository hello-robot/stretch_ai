# Simple API

The simple API allows you to give commands to the robot and stream data from the robot. It is designed to be easy to use and understand, and is a good starting point for developing more complex applications.

## Getting Started

After following the installation instructions, start the server on your robot:

```bash
ros2 launch stretch_ros2_bridge server.launch.py
```

Make sure the core test app runs:

```python
python -m stretch.app.view_images --robot_ip $ROBOT_IP
```

You should see windows popping up with camera viers from the robot, and the arm should move into a default position. The head should also turn to face the robot hand. If this all happens, you are good to go! Press `q` to quit the app.

Then, on your PC, you can easily send commands and stream data:

```python
from stretch.agent import RobotClient
robot = RobotClient(robot_ip="192.168.1.15")  # Replace with your robot's IP
# On future connection attempts, the IP address can be left blank

# Turn head towards robot's hand
robot.move_to_manip_posture()

# Move forward 0.1 along robot x axis in maniplation mode, and move arm to 0.5 meter height
robot.arm_to([0.1, 0.5, 0, 0, 0, 0])

# Turn head towards robot's base and switch base to navigation mode
# In navigation mode, we can stream velocity commands to the base for smooth motions, and base
# rotations are enabled
robot.move_to_nav_posture()

# Move the robot back to origin
# move_base_to() is only allowed in navigation mode
robot.move_base_to([0, 0, 0])

# Move the robot 0.5m forward
robot.move_base_to([0.5, 0, 0], relative=True)

# Rotate the robot 90 degrees to the left
robot.move_base_to([0, 0, 3.14159/2], relative=True)

# And to the right
robot.move_base_to([0, 0, -3.14159/2], relative=True)
```

You can find a version of this code in [examples/simple_motions.py](../examples/simple_motions.py).
