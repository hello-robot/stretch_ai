from stretch.agent.managed_operation import ManagedOperation


class GoToNavOperation(ManagedOperation):
    """Put the robot into navigation mode"""

    def can_start(self) -> bool:
        self.attempt("will switch to navigation mode.")
        return True

    def run(self) -> None:
        self.intro("Switching to navigation mode.")
        self.robot.move_to_nav_posture()

    def was_successful(self) -> bool:
        return self.robot.in_navigation_mode()
