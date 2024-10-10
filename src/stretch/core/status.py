# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Description: This file contains the status of the nodes in a LLM task plan
IDLE = "idle"
RUNNING = "running"
EXPLORING = "exploring"
FAILED = "failed"
SUCCEEDED = "succeeded"
CANNOT_START = "cannot_start"
WAITING = "waiting"
EXPLORATION_IMPOSSIBLE = "exploration_impossible"

# Some other ones
RETRYING = "retrying"
RETRYING_FAILED = "retrying_failed"
RETRYING_SUCCEEDED = "retrying_succeeded"
RETRYING_CANNOT_START = "retrying_cannot_start"
