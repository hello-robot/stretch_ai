# Stretch AI Apps

Stretch AI comes with several apps that you can run to test the robot's capabilities. These apps are designed to be easy to use and demonstrate the robot's capabilities in various scenarios.

- [Keyboard Teleop](#keyboard-teleop) - Teleoperate the robot with the keyboard.
- [Print Joint States](#print-joint-states) - Print the joint states of the robot.
- [View Images](#visualization-and-streaming-video) - View images from the robot's cameras.
- [Show Point Cloud](#show-point-cloud) - Show a joint point cloud from the end effector and head cameras.
- [Gripper](#use-the-gripper) - Open and close the gripper.
- [Rerun](#rerun) - Start a [rerun.io](https://rerun.io/)-based web server to visualize data from your robot.
- [LLM Voice Chat](#voice-chat) - Chat with the robot using LLMs.

Advanced:

- [Automatic 3d Mapping](#automatic-3d-mapping) - Automatically explore and map a room, saving the result as a PKL file.
- [Read saved map](#voxel-map-visualization) - Read a saved map and visualize it.
- [Pickup Objects](#pickup-toys) - Have the robot pickup toys and put them in a box.

Finally:

- [Dex Teleop data collection](#dex-teleop-for-data-collection) - Dexterously teleoperate the robot to collect demonstration data.
- [Learning from Demonstration (LfD)](docs/learning_from_demonstration.md) - Train SOTA policies using [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

There are also some apps for [debugging](docs/debug.md).

## List of Apps

Stretch AI is a collection of tools and applications for the Stretch robot. These tools are designed to be run on the robot itself, or on a remote computer connected to the robot. The tools are designed to be run from the command line, and are organized as Python modules. You can run them with `python -m stretch.app.<app_name>`.

Some, like `print_joint_states`, are simple tools that print out information about the robot. Others, like `mapping`, are more complex and involve the robot moving around and interacting with its environment.

All of these take the `--robot_ip` flag to specify the robot's IP address. You should only need to do this the first time you run an app for a particular IP address; the app will save the IP address in a configuration file at `~/.stretch/robot_ip.txt`. For example:

```bash
export ROBOT_IP=192.168.1.15
python -m stretch.app.print_joint_states --robot_ip $ROBOT_IP
```

### Debugging Tools

#### Keyboard Teleop

Use the WASD keys to move the robot around.

```bash
python -m stretch.app.keyboard_teleop --robot_ip $ROBOT_IP

# You may also run in a headless mode without the OpenCV gui
python -m stretch.app.keyboard_teleop --headless
```

Remember, you should only need to provide the IP address the first time you run any app from a particular endpoint (e.g., your laptop).

#### Print Joint States

To make sure the robot is connected or debug particular behaviors, you can print the joint states of the robot with the `print_joint_states` tool:

```bash
python -m stretch.app.print_joint_states --robot_ip $ROBOT_IP
```

You can also print out just one specific joint. For example, to just get arm extension in a loop, run:

```
python -m stretch.app.print_joint_states --joint arm
```

#### Visualization and Streaming Video

Visualize output from the caneras and other sensors on the robot. This will open multiple windows with wrist camera and both low and high resolution head camera feeds.

```bash
python -m stretch.app.view_images --robot_ip $ROBOT_IP
```

You can also visualize it with semantic segmentation (defaults to [Detic](https://github.com/facebookresearch/Detic/):

```bash
python -m stretch.app.view_images --robot_ip $ROBOT_IP ----run_semantic_segmentation
```

You can visualize gripper Aruco markers as well; the aruco markers can be used to determine the finger locations in the image.

```bash
python -m stretch.app.view_images --robot_ip $ROBOT_IP --aruco
```

#### Show Point Cloud

Show a joint point cloud from the end effector and head cameras. This will open an Open3d window with the point cloud, aggregated between the two cameras and displayed in world frame. It will additionally show the map's origin with a small coordinate axis; the blue arrow points up (z), the red arrow points right (x), and the green arrow points forward (y).

```bash
python -m stretch.app.show_point_cloud
```

You can use the `--reset` flag to put the robot into its default manipulation posture on the origin (0, 0, 0). Note that this is a blind, unsafe motion! Use with care.

```bash
python -m stretch.app.show_point_cloud --reset
```

#### Use the Gripper

Open and close the gripper:

```
python -m stretch.app.gripper --robot_ip $ROBOT_IP --open
python -m stretch.app.gripper --robot_ip $ROBOT_IP --close
```

Alternately:

```
python -m stretch.app.open_gripper --robot_ip $ROBOT_IP
python -m stretch.app.close_gripper --robot_ip $ROBOT_IP
```

#### Rerun Web Server

We provide the tools to publish information from the robot to a [Rerun](https://rerun.io/) web server. This is run automatically with our other apps, but if you want to just run the web server, you can do so with:

```bash
python -m stretch.app.rerun --robot_ip $ROBOT_IP
```

You should see something like this:

```
[2024-07-29T17:58:34Z INFO  re_ws_comms::server] Hosting a WebSocket server on ws://localhost:9877. You can connect to this with a native viewer (`rerun ws://localhost:9877`) or the web viewer (with `?url=ws://localhost:9877`).
[2024-07-29T17:58:34Z INFO  re_sdk::web_viewer] Hosting a web-viewer at http://localhost:9090?url=ws://localhost:9877
```

### Voice Chat

Chat with the robot using LLMs.

```bash
python -m stretch.app.voice_chat
```

### Dex Teleop for Data Collection

Dex teleop is a low-cost system for providing user demonstrations of dexterous skills right on your Stretch. This app requires the use of the [dex teleop kit](https://hello-robot.com/stretch-dex-teleop-kit).

You need to install mediapipe for hand tracking:

```
python -m pip install mediapipe
```

```bash
python -m stretch.app.dex_teleop.ros2_leader -i $ROBOT_IP --teleop-mode base_x --save-images --record-success --task-name default_task
```

[Read the data collection documentation](docs/data_collection.md) for more details.

After this, [read the learning from demonstration instructions](docs/learning_from_demonstration.md) to train a policy.

### Automatic 3d Mapping

```bash
python -m stretch.app.mapping
```

You can show visualizations with:

```bash
python -m stretch.app.mapping --show-intermediate-maps --show-final-map
```

The flag `--show-intermediate-maps` shows the 3d map after each large motion (waypoint reached), and `--show-final-map` shows the final map after exploration is done.

It will record a PCD/PKL file which can be interpreted with the `read_map` script; see below.

Another useful flag when testing is the `--reset` flag, which will reset the robot to the starting position of (0, 0, 0). This is done blindly before any execution or mapping, so be careful!

### Voxel Map Visualization

You can test the voxel code on a captured pickle file. We recommend trying with the included [hq_small.pkl](src/test/mapping/hq_small.pkl)  or [hq_large](src/test/mapping/hq_large.pkl) files, which contain a short and a long captured trajectory from Hello Robot.

```bash
python -m stretch.app.read_map -i hq_small.pkl
```

Optional open3d visualization of the scene:

```bash
python -m stretch.app.read_map -i hq_small.pkl  --show-svm
```

You can visualize instances in the voxel map with the `--show-instances` flag:

```bash
python -m stretch.app.read_map -i hq_small.pkl  --show-instances
```

You can also re-run perception with the `--run-segmentation` flag and provide a new export file with the `--export` flag:

```bash
 python -m stretch.app.read_map -i hq_small.pkl --export hq_small_v2.pkl --run-segmentation
```

You can test motion planning, frontier exploration, etc., as well. Use the `--start` flag to set the robot's starting position:

```bash
# Test motion planning
python -m stretch.app.read_map -i hq_small.pkl --test-planning --start 4.5,1.3,2.1
# Test planning to frontiers with current parameters file
python -m stretch.app.read_map -i hq_small.pkl --test-plan-to-frontier --start 4.0,1.4,0.0
# Test sampling movement to objects
python -m stretch.app.read_map -i hq_small.pkl --test-sampling --start 4.5,1.4,0.0
# Test removing an object from the map
python -m stretch.app.read_map -i hq_small.pkl --test-remove --show-instances --query "cardboard box"
```

### Pickup Objects

This will have the robot move around the room, explore, and pickup toys in order to put them in a box.

```bash
python -m stretch.app.pickup --target_object toy
```

You can add the `--reset` flag to make it go back to the start position. The default object is "toy", but you can specify other objects as well, like "bottle", "cup", or "shoe".

```
python -m stretch.app.pickup --reset
```

## Experimental

### VLM Planning

This is an experimental app that uses the voxel map to plan a path to a goal. It is not yet fully functional.

```bash
python -m stretch.app.vlm_planning
```
