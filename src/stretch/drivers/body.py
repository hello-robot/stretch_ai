# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import stretch_body.robot


class Body:
    def __init__(self):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        is_runstopped = self.robot.status["pimu"]["runstop_event"]
        if is_runstopped:
            self.robot.pimu.runstop_event_reset()
            self.robot.push_command()

    def home(self):
        if not self.robot.is_homed():
            self.robot.home()

    def get_status(self):
        status = {}
        body_status = self.robot.get_status()

        # power, IO, etc. state
        other_status = {
            "timestamp": time.time(),
            "voltage": float(body_status["pimu"]["voltage"]),
            "current": float(body_status["pimu"]["current"]),
            "is_charge_port_detecting_plug": bool(body_status["pimu"]["charger_connected"]),
            "is_charging": bool(body_status["pimu"]["charger_is_charging"]),
            "is_runstopped": bool(body_status["pimu"]["runstop_event"]),
            "is_homed": bool(self.robot.is_homed()),
        }
        status["other"] = other_status

        # mobile base state
        base_status = {
            "translational_velocity": float(body_status["base"]["x_vel"]),
            "rotational_velocity": float(body_status["base"]["theta_vel"]),
            "is_tracking": bool(body_status["base"]["left_wheel"]["is_mg_moving"])
            or bool(body_status["base"]["right_wheel"]["is_mg_moving"]),
        }
        status["mobile_base"] = base_status

        # Return abridged status if robot isn't homed
        if not self.robot.is_homed():
            return status

        # stepper joints state
        arm_status = {
            "position": float(body_status["arm"]["pos"]),
            "velocity": float(body_status["arm"]["vel"]),
            "effort": float(body_status["arm"]["motor"]["effort_pct"]),
            "num_contacts": int(body_status["arm"]["motor"]["guarded_event"]),
            "is_tracking": bool(body_status["arm"]["motor"]["is_mg_moving"]),
            "upper_limit": float(self.robot.arm.soft_motion_limits["hard"][1]),
            "lower_limit": float(self.robot.arm.soft_motion_limits["hard"][0]),
        }
        status["joint_arm"] = arm_status
        lift_status = {
            "position": float(body_status["lift"]["pos"]),
            "velocity": float(body_status["lift"]["vel"]),
            "effort": float(body_status["lift"]["motor"]["effort_pct"]),
            "num_contacts": int(body_status["lift"]["motor"]["guarded_event"]),
            "is_tracking": bool(body_status["lift"]["motor"]["is_mg_moving"]),
            "upper_limit": float(self.robot.lift.soft_motion_limits["hard"][1]),
            "lower_limit": float(self.robot.lift.soft_motion_limits["hard"][0]),
        }
        status["joint_lift"] = lift_status

        # head joints state
        head_pan_status = {
            "position": float(body_status["head"]["head_pan"]["pos"]),
            "velocity": float(body_status["head"]["head_pan"]["vel"]),
            "effort": -1 * float(body_status["head"]["head_pan"]["effort"]),
            "is_tracking": bool(
                self.robot.head.get_joint("head_pan").motor.get_moving_status() & (1 << 0) == 0
            ),
            "upper_limit": float(
                self.robot.head.get_joint("head_pan").soft_motion_limits["hard"][1]
            ),
            "lower_limit": float(
                self.robot.head.get_joint("head_pan").soft_motion_limits["hard"][0]
            ),
        }
        status["joint_head_pan"] = head_pan_status
        head_tilt_status = {
            "position": float(body_status["head"]["head_tilt"]["pos"]),
            "velocity": float(body_status["head"]["head_tilt"]["vel"]),
            "effort": -1 * float(body_status["head"]["head_tilt"]["effort"]),
            "is_tracking": bool(
                self.robot.head.get_joint("head_tilt").motor.get_moving_status() & (1 << 0) == 0
            ),
            "upper_limit": float(
                self.robot.head.get_joint("head_tilt").soft_motion_limits["hard"][1]
            ),
            "lower_limit": float(
                self.robot.head.get_joint("head_tilt").soft_motion_limits["hard"][0]
            ),
        }
        status["joint_head_tilt"] = head_tilt_status

        # wrist joints state
        wrist_yaw_status = {
            "position": float(body_status["end_of_arm"]["wrist_yaw"]["pos"]),
            "velocity": float(body_status["end_of_arm"]["wrist_yaw"]["vel"]),
            "effort": -1 * float(body_status["end_of_arm"]["wrist_yaw"]["effort"]),
            "is_tracking": bool(
                self.robot.end_of_arm.get_joint("wrist_yaw").motor.get_moving_status() & (1 << 0)
                == 0
            ),
            "upper_limit": float(
                self.robot.end_of_arm.get_joint("wrist_yaw").soft_motion_limits["hard"][1]
            ),
            "lower_limit": float(
                self.robot.end_of_arm.get_joint("wrist_yaw").soft_motion_limits["hard"][0]
            ),
        }
        status["joint_wrist_yaw"] = wrist_yaw_status
        wrist_pitch_status = {
            "position": float(body_status["end_of_arm"]["wrist_pitch"]["pos"]),
            "velocity": float(body_status["end_of_arm"]["wrist_pitch"]["vel"]),
            "effort": -1 * float(body_status["end_of_arm"]["wrist_pitch"]["effort"]),
            "is_tracking": bool(
                self.robot.end_of_arm.get_joint("wrist_pitch").motor.get_moving_status() & (1 << 0)
                == 0
            ),
            "upper_limit": float(
                self.robot.end_of_arm.get_joint("wrist_pitch").soft_motion_limits["hard"][1]
            ),
            "lower_limit": float(
                self.robot.end_of_arm.get_joint("wrist_pitch").soft_motion_limits["hard"][0]
            ),
        }
        if (
            (wrist_pitch_status["position"] > wrist_pitch_status["upper_limit"] - 1e-1)
            and (wrist_pitch_status["is_tracking"])
            and (body_status["end_of_arm"]["wrist_pitch"]["stalled"])
        ):
            wrist_pitch_status["is_tracking"] = False
        status["joint_wrist_pitch"] = wrist_pitch_status
        wrist_roll_status = {
            "position": float(body_status["end_of_arm"]["wrist_roll"]["pos"]),
            "velocity": float(body_status["end_of_arm"]["wrist_roll"]["vel"]),
            "effort": float(
                body_status["end_of_arm"]["wrist_roll"]["effort"]
            ),  # doesn't need to be multiplied by -1
            "is_tracking": bool(
                self.robot.end_of_arm.get_joint("wrist_roll").motor.get_moving_status() & (1 << 0)
                == 0
            ),
            "upper_limit": float(
                self.robot.end_of_arm.get_joint("wrist_roll").soft_motion_limits["hard"][1]
            ),
            "lower_limit": float(
                self.robot.end_of_arm.get_joint("wrist_roll").soft_motion_limits["hard"][0]
            ),
        }
        status["joint_wrist_roll"] = wrist_roll_status

        # gripper joint state
        range_rad = self.robot.end_of_arm.get_joint("stretch_gripper").soft_motion_limits["hard"]
        range_robotis = (
            self.robot.end_of_arm.motors["stretch_gripper"].world_rad_to_pct(range_rad[0]),
            self.robot.end_of_arm.motors["stretch_gripper"].world_rad_to_pct(range_rad[1]),
        )
        range_finger_rad = (
            self.robot.end_of_arm.motors["stretch_gripper"].gripper_conversion.robotis_to_finger(
                range_robotis[0]
            ),
            self.robot.end_of_arm.motors["stretch_gripper"].gripper_conversion.robotis_to_finger(
                range_robotis[1]
            ),
        )
        gripper_status = {
            "position": float(
                body_status["end_of_arm"]["stretch_gripper"]["gripper_conversion"]["finger_rad"]
            ),
            "velocity": float(
                body_status["end_of_arm"]["stretch_gripper"]["gripper_conversion"]["finger_vel"]
            ),
            "effort": float(
                body_status["end_of_arm"]["stretch_gripper"]["gripper_conversion"]["finger_effort"]
            ),
            "is_tracking": bool(
                self.robot.end_of_arm.get_joint("stretch_gripper").motor.get_moving_status()
                & (1 << 0)
                == 0
            ),
            "upper_limit": float(range_finger_rad[1]),
            "lower_limit": float(range_finger_rad[0]),
        }
        if (
            (gripper_status["position"] < gripper_status["lower_limit"] + 1e-1)
            and (gripper_status["is_tracking"])
            and (body_status["end_of_arm"]["wrist_pitch"]["stalled"])
        ):
            gripper_status["is_tracking"] = False
        status["joint_gripper"] = gripper_status

        return status

    def move_by(self, pose):
        if not self.robot.is_homed():
            return "Rejected: Robot not homed"

        if "joint_translate" in pose:
            self.robot.base.translate_by(pose["joint_translate"])
        if "joint_rotate" in pose:
            self.robot.base.rotate_by(pose["joint_rotate"])
        if "joint_lift" in pose:
            self.robot.lift.move_by(pose["joint_lift"])
        if "joint_arm" in pose:
            self.robot.arm.move_by(pose["joint_arm"])
        if "wrist_extension" in pose:
            self.robot.arm.move_by(pose["wrist_extension"])
        self.robot.push_command()
        if "joint_wrist_yaw" in pose:
            self.robot.end_of_arm.move_by("wrist_yaw", pose["joint_wrist_yaw"])
        if "joint_wrist_pitch" in pose:
            self.robot.end_of_arm.move_by("wrist_pitch", pose["joint_wrist_pitch"])
        if "joint_wrist_roll" in pose:
            self.robot.end_of_arm.move_by("wrist_roll", pose["joint_wrist_roll"])
        if "joint_gripper" in pose:
            robotis_delta = self.robot.end_of_arm.get_joint(
                "stretch_gripper"
            ).gripper_conversion.finger_to_robotis(pose["joint_gripper"])
            self.robot.end_of_arm.move_by("stretch_gripper", robotis_delta)
        if "joint_head_pan" in pose:
            self.robot.head.move_by("head_pan", pose["joint_head_pan"])
        if "joint_head_tilt" in pose:
            self.robot.head.move_by("head_tilt", pose["joint_head_tilt"])

        return "Accepted"

    def drive(self, twist):
        v = twist["translational_vel"] if "translational_vel" in twist else 0.0
        w = twist["rotational_vel"] if "rotational_vel" in twist else 0.0
        self.robot.base.set_velocity(v, w)
        self.robot.push_command()
