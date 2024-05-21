import abc
from typing import Optional


class Operation:
    pass


class Operation(abc.ABC):
    def __init__(
        self,
        name: str,
        parent: Optional[Operation] = None,
        on_success: Optional[Operation] = None,
        on_failure: Optional[Operation] = None,
        on_cannot_start: Optional[Operation] = None,
    ) -> None:
        self.name = name
        self._started = False
        self.parent = parent
        self.on_cannot_start = on_cannot_start
        self.on_success = on_success
        self.on_failure = on_failure

    @abc.abstractmethod
    def can_start(self) -> bool:
        """Returns true if the operation can begin or not."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> None:
        """Evaluate a particular operation. This will result in some changes to the world. Should set self._started to True."""
        self._started = True
        raise NotImplementedError

    def started(self) -> bool:
        """Return whether the operation has started."""
        return self._started

    @abc.abstractmethod
    def was_successful(self) -> bool:
        """Return whether the operation was successful."""
        raise NotImplementedError


class Task:
    """A task is a series of operations that are executed in sequence. At each step, we check validity."""

    def __init__(self):
        self.current_operation = None
        self.initial_operation = None
        self._all_operations = []

    def add_operation(self, operation):
        """Add this operation into the task."""
        # We will set the initial operation if not there
        if self.operation.parent is None:
            if self.initial_operation is not None:
                raise ValueError("Cannot have more than one initial operation.")
            self.initial_operation = operation
        # Add it to the list
        self._all_operations.append(operation)

    def start(self):
        """Start the task. This is a blocking loop which will continue until there are no operations left to execute."""
        self.current_operation = self.initial_operation
        if self.current_operation is None:
            raise ValueError("No initial operation set.")
        while self.current_operation is not None:
            if self.current_operation.can_start():
                self.current_operation.run()
            else:
                print(f"Operation {self.current_operation.name} cannot start.")
                if self.current_operation.on_cannot_start is None:
                    raise ValueError("Cannot start critical operation.")
                self.current_operation = self.current_operation.on_cannot_start

            if self.current_operation.was_successful():
                # Transition if we were successful
                self.current_operation = self.current_operation.on_success
            else:
                # And if we failed
                self.current_operation = self.current_operation.on_failure

        print("Task complete.")
