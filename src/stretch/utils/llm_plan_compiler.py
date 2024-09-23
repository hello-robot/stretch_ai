# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import ast

from stretch.agent.operations import (
    GoToOperation,
    GraspObjectOperation,
    PreGraspObjectOperation,
    SpeakOperation,
    WaveOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task


class LLMTreeNode:
    """Represents a node in the tree of function calls.
    Each node has a function call and two branches for success and failure"""

    def __init__(self, function_call, success=None, failure=None):
        self.function_call = function_call
        self.success = success
        self.failure = failure


class LLMPlanCompiler(ast.NodeVisitor):
    def __init__(self, agent: RobotAgent, llm_plan: str):
        self.agent = agent
        self.robot = agent.robot
        self.llm_plan = llm_plan
        self.task = None
        self.root = None

    def go_to(self, location: str):
        """Adds a GoToNavOperation to the task"""
        go_to = GoToOperation(name="go_to_" + location, agent=self.agent, robot=self.robot)
        go_to.configure(location=location)
        self.task.add_operation(go_to, True)
        return "go_to_" + location

    def pick(self, object_name: str):
        """Adds a GraspObjectOperation to the task"""
        _, self.agent.current_object = self.agent.get_instance_from_text(object_name)

        pregrasp_object = PreGraspObjectOperation(
            name="pregrasp_" + object_name,
            agent=self.agent,
            robot=self.robot,
            on_failure=None,
            retry_on_failure=True,
        )

        grasp_object = GraspObjectOperation(
            name="pick_" + object_name,
            agent=self.agent,
            robot=self.robot,
            on_failure=pregrasp_object,
        )
        grasp_object.configure(target_object=object_name, show_object_to_grasp=True)
        grasp_object.set_target_object_class(object_name)

        self.task.add_operation(pregrasp_object, True)
        self.task.add_operation(grasp_object, True)
        return ("pregrasp_" + object_name, "pick_" + object_name)

    def place(self, object_name: str):
        """Adds a PlaceObjectOperation to the task"""
        speak_not_implemented = SpeakOperation(
            name="place_" + object_name, agent=self.agent, robot=self.robot
        )
        speak_not_implemented.configure(message="Place operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "place_" + object_name

    def say(self, message: str):
        """Adds a SpeakOperation to the task"""
        say_operation = SpeakOperation(name="say_" + message, agent=self.agent, robot=self.robot)
        say_operation.configure(message=message)
        self.task.add_operation(say_operation, True)
        return "say_" + message

    def wave(self):
        """Adds a WaveOperation to the task"""
        self.task.add_operation(
            WaveOperation(name="wave", agent=self.agent, robot=self.robot), True
        )
        return "wave"

    def open_cabinet(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="open_cabinet", agent=self.agent, robot=self.robot
        )
        speak_not_implemented.configure(message="Open cabinet operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "open_cabinet"

    def close_cabinet(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="close_cabinet", agent=self.agent, robot=self.robot
        )
        speak_not_implemented.configure(message="Close cabinet operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "close_cabinet"

    def get_detections(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="get_detections", agent=self.agent, robot=self.robot
        )
        speak_not_implemented.configure(message="Get detections operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "get_detections"

    def build_tree(self, node):
        """Recursively build a tree of function calls"""
        if isinstance(node, ast.If):
            # Extract function call in the test condition
            test = node.test
            if isinstance(test, ast.Call):
                function_call = ast.unparse(test)
            else:
                raise ValueError("Unexpected test condition")

            # Create the root node with the function call
            if self.root is None:
                self.root = LLMTreeNode(function_call=function_call)
                new_node = self.root
            else:
                new_node = LLMTreeNode(function_call=function_call)

            # Recursively build success and failure branches
            if len(node.body) > 0:
                new_node.success = self.build_tree(node.body[0])
            if len(node.orelse) > 0:
                new_node.failure = self.build_tree(node.orelse[0])

            return new_node

        elif isinstance(node, ast.Expr):
            # Extract function call
            expr = node.value
            if isinstance(expr, ast.Call):
                function_call = ast.unparse(expr)
                if self.root is None:
                    self.root = LLMTreeNode(function_call=function_call)
                    return self.root
                else:
                    return LLMTreeNode(function_call=function_call)

        elif isinstance(node, ast.Module):
            # Start processing the body of the module
            if len(node.body) > 0:
                return self.build_tree(node.body[0])

        elif isinstance(node, ast.FunctionDef):
            if len(node.body) > 0:
                previous_operation = None
                first_operation = None
                for expr in node.body:
                    operation = self.build_tree(expr)

                    if first_operation is None:
                        first_operation = operation

                    if previous_operation is None:
                        previous_operation = operation
                        continue

                    if previous_operation.function_call.startswith(
                        "say"
                    ) or previous_operation.function_call.startswith("wave"):
                        previous_operation.success = operation

                    previous_operation = operation

                return first_operation
        else:
            print("Unknown node type")

        raise ValueError("Unexpected AST node")

    def convert_to_task(
        self, root: LLMTreeNode, parent_operation_name: str = None, success: bool = True
    ):
        """Recursively convert the tree into a task by adding operations and connecting them"""
        if root is None:
            return

        # Create the operation
        operation_ret = eval("self." + root.function_call)

        itermediate_operation_name = None

        if type(operation_ret) is tuple:
            root_operation_name = operation_ret[1]
            itermediate_operation_name = operation_ret[0]
        else:
            root_operation_name = operation_ret

        # root_operation_name

        # Connect the operation to the parent
        if parent_operation_name is not None:
            if success:
                # self.task.connect_on_success(parent_operation_name, root_operation_name)
                if itermediate_operation_name is not None:
                    self.task.connect_on_success(parent_operation_name, itermediate_operation_name)
                    self.task.connect_on_success(itermediate_operation_name, root_operation_name)

                else:
                    self.task.connect_on_success(parent_operation_name, root_operation_name)
            else:
                # self.task.connect_on_failure(parent_operation_name, root_operation_name)
                if itermediate_operation_name is not None:
                    self.task.connect_on_failure(parent_operation_name, itermediate_operation_name)
                    self.task.connect_on_success(itermediate_operation_name, root_operation_name)
                else:
                    self.task.connect_on_failure(parent_operation_name, root_operation_name)

        # Recursively process the success and failure branches
        self.convert_to_task(root.success, root_operation_name, True)
        self.convert_to_task(root.failure, root_operation_name, False)

    def compile(self):
        self.task = Task()
        tree = ast.parse(self.llm_plan)
        self.build_tree(tree)
        self.convert_to_task(self.root)

        return self.task
