# Debug

### Apps for Debugging

- [Test Timing](#test-timing) - Test the timing of the robot's control loop over the network.
- [Camera Info](#camera-info) - Print out camera information.

#### Test Timing

Test the timing of the robot's control loop over the network. This will print out the time it takes to send a command to the robot and receive a response. It will show a histogram after a fixed number of iterations given by the `--iterations` flag (default is 500).

```bash
python -m stretch.app.timing --robot_ip $ROBOT_IP

# Headless mode - no display
python -m stretch.app.timing --headless

# Set the number of iterations per histogram to 1000
python -m stretch.app.timing --iterations 1000
```

#### Camera Info

Print out information about the cameras on the robot for debugging purposes:

```bash
python -m stretch.app.debug.camera_info --robot_ip $ROBOT_IP
```

This will print out information about the resolutions of different images sent by the robot's cameras, and should show something like this:

```
---------------------- Camera Info ----------------------
Servo Head RGB shape: (320, 240, 3) Servo Head Depth shape: (320, 240)
Servo EE RGB shape: (240, 320, 3) Servo EE Depth shape: (240, 320)
Observation RGB shape: (640, 480, 3) Observation Depth shape: (640, 480)
```
