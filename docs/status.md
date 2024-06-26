# Status

You can fetch the robot's status using:

```python
import stretch
stretch.connect()
status = stretch.get_status()
```

If you were to print out `status`, you'd see a large Python dictionary. In this document, we'll cover the structure of this status dictionary.

## Top Level Keys

The top level keys of `status` will be:

- other
- mobile_base
- joint_arm
- joint_lift
- a whole bunch more joints...

To see what joints are on the robot, check out the [Stretch Hardware Overview](https://docs.hello-robot.com/0.3/getting_started/stretch_hardware_overview/).

## Unhomed robot

Each joint on the robot "wakes up" (i.e. is powered on) not knowing how far open/closed the joint is within its joint range. This is because Stretch uses relative joint encoders. A simple 30 second homing procedure allows each joint to identify where zero is located and the joints' position within its joint range.

But, we can't provide joint status until the robot is homed. Therefore, if the robot isn't homed, you'll only see the following top level keys:

- other
- mobile_base - the mobile base never needs homing

## Other section

```python
print(status['other'])

# example output:
# {'timestamp': 1714706612.7522728, 'voltage': 13.748682737350464, 'current': 4.078564992440598,
#  'is_charge_port_detecting_plug': True, 'is_charging': True, 'is_runstopped': False, 'is_homed': False}
```

The "other" section of the status dictionary includes the following keys:

- **timestamp**: when the status was collected
- **voltage**: the battery voltage of the robot (\< 10.5V is low battery)
- **current**: the battery current of the robot
- **is_charge_port_detecting_plug**: whether the mechanical switch within the charge port detects that a charger plug is plugged in
- **is_charging**: whether the robot is actually receiving charge. Combined with is_charge_port_detecting_plug, you can warn users that they've plugged in a faulty or turned-off charger when a plug is detected but no charge is being received.
- **is_runstopped**: whether the robot is in a "run-stop", which is a safety mode where all joints are frozen and backdrive-able. There's a big easy-to-press run-stop button on Stretch so users may stop the robot's motion easily.
- **is_homed**: whether or not the robot is homed

## Mobile base section

```python
print(status['mobile_base'])

# example output:
# {'translational_velocity': 0.0, 'rotational_velocity': 0.0, 'is_tracking': False}
```

The keys are:

- **translational_velocity**: the forward/backwards velocity of the robot, as tracked by wheel odometry
- **rotational_velocity**: the in-place rotation velocity of the robot, as tracked by wheel odometry
- **is_tracking**: whether the mobile base is tracking a user command

## Joint sections

```python
print(status['joint_arm'])

# example output:
# {'position': 0.00014485202786421945, 'velocity': 5.7101868488658656e-05, 'effort': -4.39623047631315e-45,
#  'num_contacts': 17, 'is_tracking': False, 'upper_limit': 0.5197662863936018, 'lower_limit': 0.0}
```

The keys for each joint are the same. They are:

- **position**: the joint position, in meters for prismatic joints, in radians for revolute joints
- **velocity**: the joint velocity, in m/s or rad/s
- **effort**: the joint effort being applied, in percentage from 0%-100% for positive effort, or -100%-0% for negative effort
- **num_contacts**: the arm/lift joints are capable of detecting contact, and will report the number of contacts detected
- **is_tracking**: whether the joint in tracking a user command
- **upper_limit**: the upper limit of the joint range, in meters or radians
- **lower_limit**: the lower limit of the joint range, in meters or radians. See the [Stretch's Joint Limits](https://docs.hello-robot.com/0.3/python/moving/#querying-stretchs-joint-limits) guide to see the joints ranges visually.

```python
print(status['joint_wrist_yaw']['position'])

# example output:
# 2.9950974883467145
```
