from stretch.agent.base import ManagedOperation


class GoToNavOperation(ManagedOperation):
    """Put the robot into navigation mode"""

    def can_start(self) -> bool:
        self.attempt("will switch to navigation mode.")
        return True

    def run(self) -> None:
        self.intro("Switching to navigation mode.")
        self.robot.move_to_nav_posture()
        self.info("Robot is in navigation mode: {}".format(self.robot.in_navigation_mode()))
        print("Robot is in navigation mode: {}".format(self.robot._control_mode))

    def was_successful(self) -> bool:
        res = self.robot.in_navigation_mode()
        if res:
            self.cheer("Robot is in navigation mode.")
        else:
            self.error("Robot is still not in navigation mode.")
        return res
