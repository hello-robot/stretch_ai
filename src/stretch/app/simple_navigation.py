# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent import RobotClient

print("Initialize the robot client")
print("robot = RobotClient()")
robot = RobotClient()
print("Done")
# On future connection attempts, the IP address can be left blank

# Turn head towards robot's hand
print("Turn head towards robot's hand")
print("robot.move_to_manip_posture()")
robot.move_to_manip_posture()
print("Done")

# Move forward 0.1 along robot x axis in maniplation mode, and move arm to 0.5 meter height
print("Move forward 0.1 along robot x axis in maniplation mode, and move arm to 0.5 meter height")
print("robot.arm_to([0.1, 0.5, 0, 0, 0, 0], blocking=True)")
robot.arm_to([0.1, 0.5, 0, 0, 0, 0], blocking=True)
print("Done")

# Turn head towards robot's base and switch base to navigation mode
# In navigation mode, we can stream velocity commands to the base for smooth motions, and base
# rotations are enabled
print("Turn head towards robot's base and switch base to navigation mode")
print("robot.move_to_nav_posture()")
robot.move_to_nav_posture()
print("Done")

# Move the robot back to origin
# navigate_to() is only allowed in navigation mode
print("Move the robot back to origin")
print("robot.navigate_to([0, 0, 0])")
robot.navigate_to([0, 0, 0])
print("Done")

# Move the robot 0.5m forward
print("Move the robot 0.5m forward")
print("robot.navigate_to([0.5, 0, 0], relative=True, blocking=True)")
robot.navigate_to([0.5, 0, 0], relative=True, blocking=True)
print("Done")

# Rotate the robot 90 degrees to the left
print("Rotate the robot 90 degrees to the left")
print("robot.navigate_to([0, 0, 3.14159/2], relative=True, blocking=True)")
robot.navigate_to([0, 0, 3.14159 / 2], relative=True, blocking=True)
print("Done")

# And to the right
print("Rotate the robot 90 degrees to the right")
print("robot.navigate_to([0, 0, -3.14159/2], relative=True, blocking=True)")
robot.navigate_to([0, 0, -3.14159 / 2], relative=True, blocking=True)

print("Stop the robot")
print("robot.stop()")
robot.stop()
