# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import abc
from typing import Optional

from termcolor import colored

from stretch.utils.logger import Logger

logger = Logger(__name__)


class Operation(abc.ABC):
    """An operation is a single unit of work that can be executed. It can be part of a task. It
    also has an associated set of pre- and post-conditions. Operations can be chained together to
    form an executable task."""

    def __init__(
        self,
        name: str,
        parent: Optional["Operation"] = None,
        on_success: Optional["Operation"] = None,
        on_failure: Optional["Operation"] = None,
        on_cannot_start: Optional["Operation"] = None,
        retry_on_failure: bool = False,
    ) -> None:
        self._name = name
        self._started = False
        self.parent = parent
        self.on_cannot_start = on_cannot_start
        self.on_success = on_success
        self.on_failure = on_failure
        self.retry_on_failure = retry_on_failure

        if self.parent is not None and self.parent.on_success is None:
            self.parent.on_success = self

        # Overload failure to just retry this one
        if self.retry_on_failure:
            if self.on_failure is not None:
                raise RuntimeError(
                    f"Cannot have on_failure set for {self.name} - it will just retry itself."
                )
            self.on_failure = self

        self.status = "idle"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_status(self) -> str:
        """Return the status of the operation."""
        return self.status

    def set_status(self, status: str):
        """Set the status of the operation."""
        self.status = status

    @property
    def name(self) -> str:
        """Return the name of the operation."""
        return self._name

    def configure(self, **kwargs) -> None:
        """Configure the operation with the given keyword arguments. By default this just does nothing."""

    @abc.abstractmethod
    def can_start(self) -> bool:
        """Returns true if the operation can begin or not."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> None:
        """Evaluate a particular operation. This will result in some changes to the world. Should
        set self._started to True."""
        self._started = True
        raise NotImplementedError

    def started(self) -> bool:
        """Return whether the operation has started."""
        return self._started

    @abc.abstractmethod
    def was_successful(self) -> bool:
        """Return whether the operation was successful."""
        raise NotImplementedError

    def __call__(self, **kwargs) -> bool:
        """Run the operation."""
        self.configure(**kwargs)
        if self.can_start():
            self.run()
        else:
            logger.error(f"Operation {self.name} cannot start.")
            return False
        self.run()
        return self.was_successful()


class Task:
    """A task is a series of operations that are executed in sequence. At each step, we check validity."""

    def __init__(self):
        self.current_operation = None
        self.initial_operation = None
        self._all_operations = []
        self._terminal_operations = []
        self._operations = dict()

    def add_operation(self, operation, terminal: bool = False):
        """Add this operation into the task.

        Args:
            operation: The operation to add.
            terminal: Whether this operation will end the task plan or not.
        """

        # Add the operation
        self._operations[operation.name] = operation

        # We will set the initial operation if not there
        if self.initial_operation is None:
            if operation.parent is None:
                self.initial_operation = operation

        # Get the last operation to make it easier to add
        prev_operation = self._all_operations[-1] if len(self._all_operations) > 0 else None
        # We can add this to the list of terminal operations for the task plan
        if terminal:
            self._terminal_operations.append(operation)

        if prev_operation is not None:
            if prev_operation.on_success is None and not terminal:
                # If we have a previous operation, set the parent
                prev_operation.on_success = operation

        # Add it to the list
        self._all_operations.append(operation)

    def connect_on_failure(self, operation_name: str, on_failure_name: str):
        """Connect the operation to the on_failure operation."""
        operation = self._operations[operation_name]
        on_failure = self._operations[on_failure_name]
        operation.on_failure = on_failure

    def connect_on_cannot_start(self, operation_name: str, on_cannot_start_name: str):
        """Connect the operation to the on_cannot_start operation."""
        operation = self._operations[operation_name]
        on_cannot_start = self._operations[on_cannot_start_name]
        operation.on_cannot_start = on_cannot_start

    def connect_on_success(self, operation_name: str, on_success_name: str):
        """Connect the operation to the on_success operation."""
        operation = self._operations[operation_name]
        on_success = self._operations[on_success_name]
        operation.on_success = on_success

    def info(self, message):
        print(colored("[TASK INFO] " + str(message), "cyan"))

    def run(self):
        """Start the task. This is a blocking loop which will continue until there are no operations left to execute."""
        self.current_operation = self.initial_operation
        if self.current_operation is None:
            raise ValueError("No initial operation set.")
        while self.current_operation is not None:
            if self.current_operation.can_start():
                self.info(f"Starting operation {self.current_operation.name}")
                self.current_operation.run()
                self.info("Operation complete.")
                if self.current_operation.was_successful():
                    # Transition if we were successful
                    self.info(f"Operation {self.current_operation.name} successful.")
                    self.current_operation = self.current_operation.on_success
                    if self.current_operation is None:
                        self.info("Task completed successfully!")
                    else:
                        self.info(f"Transitioning to {self.current_operation.name}")
                else:
                    # And if we failed
                    self.info(f"Operation {self.current_operation.name} failed.")
                    self.current_operation = self.current_operation.on_failure
            else:
                self.info(f"Operation {self.current_operation.name} cannot start.")
                if self.current_operation.on_cannot_start is None:
                    raise ValueError("Cannot start critical operation.")
                self.current_operation = self.current_operation.on_cannot_start

        print("Task complete.")
