# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

from stretch.core.task import Operation, Task


class Data:
    def __init__(self, num: int = 0):
        self.num = num


class AddOperation(Operation):
    def __init__(self, data, num, **kwargs):
        super().__init__(f"+{num}", **kwargs)
        self.data = data
        self.num = num

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        print(f"Running {self.name}: {self.data.num} + {self.num} = {self.data.num + self.num}")
        self.data.num += self.num

    def was_successful(self) -> bool:
        return True


class ReturnIfEvenOperation(Operation):
    def __init__(self, data, **kwargs):
        super().__init__("Zero if even", **kwargs)
        self.data = data

    def can_start(self) -> bool:
        print(f"Checking {self.name}: {self.data.num} % 2 == 0")
        return self.data.num % 2 == 0

    def run(self) -> None:
        print(f"!!! {self.data.num} is even! !!!")

    def was_successful(self) -> bool:
        return True


def test_task():
    data = Data(0)
    task = Task()

    add1 = AddOperation(data, 1)
    add2 = AddOperation(data, 2, parent=add1)
    add3 = AddOperation(data, 3, parent=add2)
    add1b = AddOperation(data, 1)
    zero = ReturnIfEvenOperation(data, parent=add3, on_cannot_start=add1b)
    add1b.on_success = zero

    task.add_operation(add1)
    task.add_operation(add2)
    task.add_operation(add3)
    task.add_operation(add1b)
    task.add_operation(zero, terminal=True)

    task.run()
    print("From 0:", data.num)

    assert data.num == 6, "From 0, data should get transmuted into 6"

    data.num = 1
    task.run()

    assert (
        data.num == 8
    ), "From 1, data should get transmuted into 8 - loops through + 1 an extra time"


if __name__ == "__main__":
    test_task()
