# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

from stretch.agent import RobotClient

print("Initialize the robot client")
print("robot = RobotClient()")
robot = RobotClient()
print("Done")
# On future connection attempts, the IP address can be left blank

# Turn head towards robot's hand
print("Turn head towards robot's hand")
robot.say("Moving head to look at hand")
time.sleep(0.5)
print("robot.move_to_manip_posture()")
robot.move_to_manip_posture()
print("Done")

# Move forward 0.1 along robot x axis in maniplation mode, and move arm to 0.5 meter height
print("Move forward 0.1 along robot x axis in maniplation mode, and move arm to 0.5 meter height")
robot.say("Moving in manipulation mode")
time.sleep(0.5)
print("robot.arm_to([0.1, 0.5, 0, 0, 0, 0], blocking=True)")
robot.arm_to([0.1, 0.5, 0, 0, 0, 0])
print("Done")

# Turn head towards robot's base and switch base to navigation mode
# In navigation mode, we can stream velocity commands to the base for smooth motions, and base
# rotations are enabled
print("Turn head towards robot's base and switch base to navigation mode")
robot.say("Switching to navigation mode")
time.sleep(0.5)
print("robot.move_to_nav_posture()")
robot.move_to_nav_posture()
print("Done")

# Move the robot back to origin
# move_base_to() is only allowed in navigation mode
print("Move the robot back to origin")
robot.say("Moving to origin")
time.sleep(0.5)
print("robot.move_base_to([0, 0, 0])")
robot.move_base_to([0, 0, 0])
print("Done")

# Move the robot 0.5m forward
print("Move the robot 0.25m forward")
robot.say("Moving forward 0.25 meters")
time.sleep(0.5)
print("robot.move_base_to([0.25, 0, 0], relative=True)")
robot.move_base_to([0.5, 0, 0], relative=True)
print("Done")

# Rotate the robot 90 degrees to the left
print("Rotate the robot 90 degrees to the left")
robot.say("Rotating 90 degrees to the left")
time.sleep(0.5)
print("robot.move_base_to([0, 0, 3.14159/2], relative=True)")
robot.move_base_to([0, 0, 3.14159 / 2], relative=True, verbose=True)
print("Done")

# And to the right
print("Rotate the robot 90 degrees to the right")
robot.say("Rotating 90 degrees to the right")
time.sleep(0.5)
print("robot.move_base_to([0, 0, -3.14159/2], relative=True, blocking=True)")
robot.move_base_to([0, 0, -3.14159 / 2], relative=True, verbose=True)

# Move the robot back to origin
# move_base_to() is only allowed in navigation mode
print("Move the robot back to origin")
robot.say("Moving back to origin to finish")
time.sleep(0.5)
print("robot.move_base_to([0, 0, 0])")
robot.move_base_to([0, 0, 0])
print("Done")

print("Stop the robot")
robot.say("Stopping")
print("robot.stop()")
robot.stop()
