# Dex Teleop Example App

This is a modified version of [stretch dex teleop](https://github.com/hello-robot/stretch_dex_teleop) which collects data for use training [Dobb-E](https://dobb-e.com/) policies.

## Installation

Follow the [Stretch dex teleop](https://github.com/hello-robot/stretch_dex_teleop) instructions to calibrate your camera and make sure things are working.

Webcam should be plugged into your workstation or laptop (\`\`leader pc'')

## Running

### On the Robot

These steps replace the usual server *for now*.

Start the image server:

```
python -m stretch.app.dex_teleop.send_d405_images -r
```

Start the follower:

```
python -m stretch.app.dex_teleop.follower
```

### On the Leader PC

Run the leader script:

```bash
python -m stretch.app.dex_teleop.leader
```

A window should appear, showing the view from the end effector camera. This should be roughly real time; if not, improve your network connection somehow. Press space to start and stop recording demonstrations.

When collecting data, you should set task, user, and environment, instead of just using the default for all of the above. For example:

```bash
python -m stretch.app.dex_teleop.leader --task grasp_cup --user Chris --env ChrisKitchen1
```

Collect a few demonstrations per example task/environment that you want to test in.

#### Data Collection

The robot will start as soon as the Dex Teleop tool is visible! However, it will not start recording until you initiate data collection with the keyboard using the `space` key. This allows you to put the arm in a reasonable position to learn your skill.

#### Keyboard Controls

- Press `space`: start/stop recording a demonstration. It will be written to a file based on the provided task, user, and environment, with a subfolder based on the date and time.
- Press `esc`: quit the program.

#### Pausing Motions during an Episode

We provide the ability to pause the robot's motion during a recording. Running `stretch.app.dex_teleop.ros2_leader` with the `-c` flag adds the ability to "clutch" the robot's motion by holding up your other hand (the hand not holding the tongs) over the dex teleop camera. When the camera sees your other hand, it will pause tracking and allow you to adjust the pose of the tongs without moving the end effector. This is similar to lifting up and repositioning a computer mouse.
