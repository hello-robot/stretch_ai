# Debug

### Apps for Debugging

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
