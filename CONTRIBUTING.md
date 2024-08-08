# Contributing to Stretch AI

We welcome contributions to Stretch AI! Please read the following guidelines before submitting a pull request.

This repository is in an early state of development. Guidelines are subject to change and tests will be by necessity incomplete.

Before contributing, you will need to sign a [Contributor License Agreement](https://gist.github.com/hello-cpaxton/8881ca0a05858dcaee4ebf0207d92f83). For more information, you can check out the [Stretch Contributor License Agreement](https://github.com/hello-robot/stretch_contributor_license_agreements) page for more information.

We use the SAP [CLA Assistant bot](https://github.com/cla-assistant/cla-assistant), which will open a comment and give you instructions when you open a pull request.

### Setup

Install the code and set up the pre-commit hooks:

```
git clone https://github.com/hello-robot/stretch_ai.git --recursive
cd stretch_ai/src
pip install -e .[dev]
pre-commit install
```

### Style

We use [black](https://black.readthedocs.io/en/stable/) and [flake8](https://flake8.pycqa.org/en/latest/) to format our code.
In addition, we use [isort](https://pycqa.github.io/isort/) for sorting imports, [mypy](https://mypy-lang.org/) for static type checking, and [codespell](https://github.com/codespell-project/codespell) for spell checking, among other things.

You can run the pre-commit hooks on all files with:
```
pre-commit run --all-files
```

Please make sure that all changes are made and that the pre-commit hooks pass before submitting a pull request.

If you need to temporarily commit something that is not passing, use:
```
git commit --no-verify
```

### Pull Requests

We follow a squash-and-merge strategy for pull requests, which means that all commits in a PR are squashed into a single commit before merging. This keeps the git history clean and easy to read.

Please make sure your PR is up-to-date with the latest changes in the main branch before submitting. You can do this by rebasing your branch on the main branch:

```
git checkout main
git pull
git checkout <your-branch>
git rebase main
```

#### Draft PRs

If a PR is still a work-in-progress and not ready for review, please open it with "WIP: (final PR title)" in the title to indicate that it is still a work in progress. This will indicate to reviewers that the PR is not ready for review yet. In addition, use the "Draft" PR status on github to indicate that the PR is not ready yet.

### Documentation

Please make sure to update the documentation if you are adding new features or changing existing ones. This includes docstrings, README, and any other relevant documentation. Use [type hints](https://docs.python.org/3/library/typing.html) to make the code more readable and maintainable.

For example:

```python
def add(a: int, b: int) -> int:
    return a + b
```

This shows what `a` and `b` are expected to be and what the function returns -- in this case, all are `int` variables.

### Testing

We use [pytest](https://docs.pytest.org/en/7.0.1/) for testing. Please make sure to add tests for new features or changes to existing code. You can run the tests with:

```
cd src
pytest
```

### File Structure

The code is organized as follows. Inside the core package `src/stretch`:

- [core](src/stretch/core) is basic tools and interfaces
- [app](src/stretch/app)  contains individual endpoints, runnable as `python -m stretch.app.<app_name>`, such as mapping, discussed above.
- [motion](src/stretch/motion) contains motion planning tools, including [algorithms](src/stretch/motion/algo) like RRT.
- [mapping](src/stretch/mapping) is broken up into tools for voxel (3d / ok-robot style), instance mapping
- [agent](src/stretch/agent) is aggregate functionality, particularly robot_agent which includes lots of common tools including motion planning algorithms.
  - In particular, `agent/zmq_client.py` is specifically the robot control API, an implementation of the client in core/interfaces.py. there's another ROS client in `stretch_ros2_bridge`.
  - [agent/robot_agent.py](src/stretch/agent/robot_agent.py) is the main robot agent, which is a high-level interface to the robot. It is used in the `app` scripts.
  - [agent/base](src/stretch/agent/base) contains base classes for creating tasks, such as the [TaskManager](src/stretch/agent/base/task_manager.py) class and the [ManagedOperation](src/stretch/agent/base/managed_operation.py) class.
  - [agent/task](src/stretch/agent/task) contains task-specific code, such as for the `pickup` task. This is divided between "Managers" like [pickup_manager.py](src/stretch/agent/task/pickup_manager.py) which are composed of "Operations." Each operation is a composable state machine node with pre- and post-conditions.
  - [agent/operations](src/stretch/agent/operations) contains the individual operations, such as `move_to_pose.py` which moves the robot to a given pose.

The [stretch_ros2_bridge](src/stretch_ros2_bridge) package is a ROS2 bridge that allows the Stretch AI code to communicate with the ROS2 ecosystem. It is a separate package that is symlinked into the `ament_ws` workspace on the robot.
