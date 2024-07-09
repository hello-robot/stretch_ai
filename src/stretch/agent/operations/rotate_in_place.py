from stretch.agent.base import ManagedOperation


class RotateInPlaceOperation(ManagedOperation):
    """Rotate the robot in place. Number of steps is determined by parameters file."""

    def can_start(self) -> bool:
        self.attempt(f"Rotating for {self.parameters['in_place_rotation_steps']} steps.")
        return True

    def run(self) -> None:
        self.intro("rotating for {self.parameters['in_place_rotation_steps']} steps.")
        self._successful = False
        self.robot.rotate_in_place(
            steps=self.parameters["in_place_rotation_steps"],
            visualize=False,
        )
        self._successful = True

    def was_successful(self) -> bool:
        return self._successful
