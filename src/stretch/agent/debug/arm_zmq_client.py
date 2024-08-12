#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
import timeit

import click
import cv2
from home_robot.agent.multitask.zmq_client import HomeRobotZmqClient


class TestArmZmqClient(HomeRobotZmqClient):
    def blocking_spin(self, verbose: bool = False, visualize: bool = False):
        """this is just for testing"""
        sum_time = 0
        steps = 0
        t0 = timeit.default_timer()

        # For debugging
        prev_q = None
        start_t = t0
        rate = 3
        first_frame = True

        while not self._finish:

            output = self.recv_socket.recv_pyobj()
            if output is None:
                continue

            output["rgb"] = cv2.imdecode(output["rgb"], cv2.IMREAD_COLOR)
            compressed_depth = output["depth"]
            depth = cv2.imdecode(compressed_depth, cv2.IMREAD_UNCHANGED)
            output["depth"] = depth / 1000.0

            if first_frame:
                self.move_to_manip_posture(blocking=False)
                time.sleep(3.0)
                first_frame = False
            else:
                t1 = timeit.default_timer()
                step = (t1 - start_t) % 10
                if step % rate < 1:
                    # send a command
                    if step > rate:
                        # B POSITION
                        q = [0, 0.5, 0.0, 0, 0, 0]
                        if q != prev_q:
                            print(f"{q=}")
                            self.arm_to(q)
                    else:
                        # a position
                        q = [0.1, 0.75, 0.5, 0, 0, 0]
                        if q != prev_q:
                            print(f"{q=}")
                            self.arm_to(q)
                    prev_q = q

            self._update_obs(output)
            # with self._act_lock:
            #    if len(self._next_action) > 0:

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose:
                print("Control mode:", self._control_mode)
                print(f"time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}")
            t0 = timeit.default_timer()


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="192.168.1.15")
def main(
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "192.168.1.15",
):
    client = TestArmZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
    )
    client.blocking_spin(verbose=False)


if __name__ == "__main__":
    main()
