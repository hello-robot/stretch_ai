# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import ast
import re

from stretch.agent.operations import (
    GoToNavOperation,
    GraspObjectOperation,
    NavigateToObjectOperation,
    PlaceObjectOperation,
    PreGraspObjectOperation,
    SearchForObjectOnFloorOperation,
    SearchForReceptacleOperation,
    SetCurrentObjectOperation,
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


# # Dummy execute_task function
# def execute_task(go_to, pick, place, say, wave, open_cabinet, close_cabinet, get_detections):
#     return None


class LLMPlanCompiler(ast.NodeVisitor):
    def __init__(self, agent: RobotAgent, llm_plan: str):
        self.agent = agent
        self.robot = agent.robot
        self.llm_plan = llm_plan
        self.task = None
        self.root = None
        self._operation_naming_counter = 0

    def add_self_prefix(self, func_str):
        # Regular expression to match function calls
        updated_func_str = re.sub(r'(\w+)\(', r'self.\1(', func_str)
        return updated_func_str

    def go_to(self, location: str):
        """Adds a GoToNavOperation to the task"""
        # _, current_object = self.agent.get_instance_from_text(location)

        # if current_object is not None:
        #     print(f"Setting current object to {current_object}")
        #     set_current_object = SetCurrentObjectOperation(
        #         name="set_current_object_" + location + f"_{str(self._operation_naming_counter)}",
        #         agent=self.agent,
        #         robot=self.robot,
        #         target=current_object,
        #     )
        #     self._operation_naming_counter += 1
        #     self.task.add_operation(set_current_object, True)

        # go_to = NavigateToObjectOperation(
        #     name="go_to_" + location + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     robot=self.robot,
        #     to_receptacle=False,
        # )
        # self._operation_naming_counter += 1
        # go_to.configure(location=location)
        # self.task.add_operation(go_to, True)

        # if current_object is not None:
        #     self.task.connect_on_success(set_current_object.name, go_to.name)
        #     return (
        #         "set_current_object_" + location + f"_{str(self._operation_naming_counter - 2)}",
        #         "go_to_" + location + f"_{str(self._operation_naming_counter - 1)}",
        #     )
        # return "go_to_" + location + f"_{str(self._operation_naming_counter - 1)}"

        speak_go_to = SpeakOperation(
            name="speak_go_to_" + location + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_go_to.configure(message=f"Going to {location}")

        self.task = Task()
        self.task.add_operation(speak_go_to, True)
        return self.task.run()

        # return "speak_go_to_" + location + f"_{str(self._operation_naming_counter - 1)}"

    def pick(self, object_name: str):
        """Adds a GraspObjectOperation to the task"""
        # Try to expand the frontier and find an object; or just wander around for a while.

        speak_pick = SpeakOperation(
            name="speak_pick_" + object_name + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_pick.configure(message=f"Going to pick up {object_name}")

        self.task = Task()
        self.task.add_operation(speak_pick, True)
        return self.task.run()

        # go_to_navigation_mode = GoToNavOperation(
        #     name="go_to_navigation_mode" + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     retry_on_failure=True,
        # )
        # self._operation_naming_counter += 1

        # search_for_object = SearchForObjectOnFloorOperation(
        #     name=f"search_for_{object_name}_on_floor" + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     retry_on_failure=True,
        #     match_method="feature",
        #     require_receptacle=False,
        # )
        # self._operation_naming_counter += 1
        # search_for_object.set_target_object_class(object_name)

        # go_to_object = NavigateToObjectOperation(
        #     name="go_to_object" + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     parent=search_for_object,
        #     on_cannot_start=search_for_object,
        #     to_receptacle=False,
        # )
        # self._operation_naming_counter += 1

        # pregrasp_object = PreGraspObjectOperation(
        #     name="pregrasp_" + object_name + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     robot=self.robot,
        #     on_failure=None,
        #     retry_on_failure=True,
        # )
        # self._operation_naming_counter += 1

        # grasp_object = GraspObjectOperation(
        #     name="pick_" + object_name + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     robot=self.robot,
        #     on_failure=pregrasp_object,
        # )
        # self._operation_naming_counter += 1
        # grasp_object.configure(
        #     target_object=object_name, show_object_to_grasp=False, show_servo_gui=False
        # )
        # grasp_object.set_target_object_class(object_name)
        # grasp_object.servo_to_grasp = True
        # grasp_object.match_method = "feature"

        # self.task.add_operation(go_to_navigation_mode, True)
        # self.task.add_operation(search_for_object, True)
        # self.task.add_operation(go_to_object, True)
        # self.task.add_operation(pregrasp_object, True)
        # self.task.add_operation(grasp_object, True)

        # self.task.connect_on_success(go_to_navigation_mode.name, search_for_object.name)
        # self.task.connect_on_success(search_for_object.name, go_to_object.name)
        # self.task.connect_on_success(go_to_object.name, pregrasp_object.name)
        # self.task.connect_on_success(pregrasp_object.name, grasp_object.name)

        # return (
        #     go_to_navigation_mode.name,
        #     grasp_object.name,
        # )

    def place(self, receptacle_name: str):
        """Adds a PlaceObjectOperation to the task"""
        speak_place = SpeakOperation(
            name="speak_place_" + receptacle_name + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_place.configure(message=f"Going to place in {receptacle_name}")
        
        self.task = Task()
        self.task.add_operation(speak_place, True)
        return self.task.run()

        # return "speak_place_" + receptacle_name + f"_{str(self._operation_naming_counter - 1)}"
        # go_to_navigation_mode = GoToNavOperation(
        #     name="go_to_navigation_mode" + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     retry_on_failure=True,
        # )
        # self._operation_naming_counter += 1

        # search_for_receptacle = SearchForReceptacleOperation(
        #     name=f"search_for_{receptacle_name}" + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     retry_on_failure=True,
        #     match_method="feature",
        # )
        # self._operation_naming_counter += 1
        # search_for_receptacle.set_target_object_class(receptacle_name)

        # go_to_receptacle = NavigateToObjectOperation(
        #     name="go_to_receptacle" + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     parent=search_for_receptacle,
        #     on_cannot_start=search_for_receptacle,
        #     to_receptacle=True,
        # )
        # self._operation_naming_counter += 1

        # place_object_on_receptacle = PlaceObjectOperation(
        #     name="place_" + receptacle_name + f"_{str(self._operation_naming_counter)}",
        #     agent=self.agent,
        #     robot=self.robot,
        #     on_cannot_start=go_to_receptacle,
        #     require_object=True,
        # )
        # self._operation_naming_counter += 1

        # self.task.add_operation(go_to_navigation_mode, True)
        # self.task.add_operation(search_for_receptacle, True)
        # self.task.add_operation(go_to_receptacle, True)
        # self.task.add_operation(place_object_on_receptacle, True)

        # self.task.connect_on_success(go_to_navigation_mode.name, search_for_receptacle.name)
        # self.task.connect_on_success(search_for_receptacle.name, go_to_receptacle.name)
        # self.task.connect_on_success(go_to_receptacle.name, place_object_on_receptacle.name)

        # return (
        #     go_to_navigation_mode.name,
        #     place_object_on_receptacle.name,
        # )

    def say(self, message: str):
        """Adds a SpeakOperation to the task"""
        say_operation = SpeakOperation(
            name="say_" + message + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        say_operation.configure(message=message)
        # self.task.add_operation(say_operation, True)
        # return "say_" + message + f"_{str(self._operation_naming_counter - 1)}"

        self.task = Task()
        self.task.add_operation(say_operation, True)
        return self.task.run()

    def wave(self):
        """Adds a WaveOperation to the task"""
        # self.task.add_operation(
        #     WaveOperation(
        #         name="wave" + f"_{str(self._operation_naming_counter)}",
        #         agent=self.agent,
        #         robot=self.robot,
        #     ),
        #     True,
        # )
        # self._operation_naming_counter += 1
        # return "wave" + f"_{str(self._operation_naming_counter - 1)}"

        speak_wave = SpeakOperation(
            name="speak_wave" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_wave.configure(message="Waving")
        # self.task.add_operation(speak_wave, True)

        # return "speak_wave" + f"_{str(self._operation_naming_counter - 1)}"

        self.task = Task()
        self.task.add_operation(speak_wave, True)
        return self.task.run()

    def open_cabinet(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="open_cabinet" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_not_implemented.configure(message="Open cabinet operation not implemented")
        # self.task.add_operation(speak_not_implemented, True)
        # return "open_cabinet" + f"_{str(self._operation_naming_counter - 1)}"
        self.task = Task()
        self.task.add_operation(speak_not_implemented, True)
        return self.task.run()

    def close_cabinet(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="close_cabinet" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_not_implemented.configure(message="Close cabinet operation not implemented")
        # self.task.add_operation(speak_not_implemented, True)
        # return "close_cabinet" + f"_{str(self._operation_naming_counter - 1)}"
        self.task = Task()
        self.task.add_operation(speak_not_implemented, True)
        return self.task.run()

    def get_detections(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="get_detections" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_not_implemented.configure(message="Get detections operation not implemented")
        # self.task.add_operation(speak_not_implemented, True)
        # return "get_detections" + f"_{str(self._operation_naming_counter - 1)}"
        self.task = Task()
        self.task.add_operation(speak_not_implemented, True)
        return self.task.run()

    # def parse_if_block(self, node):
    #     """Parse a single if block and its nested conditions."""
    #     if isinstance(node, ast.If):
    #         condition = node.test
    #         success_node = self.build_tree(node.body)
    #         failure_node = self.build_tree(node.orelse) if node.orelse else None
    #         return LLMTreeNode(ast.unparse(node.test), success_node, failure_node)

    #     return None
    
    # def build_tree(self, nodes):
    #     """Build the tree from the list of AST nodes."""
    #     if not nodes:
    #         return None
        
    #     root = None
    #     current_node = None
    #     for node in nodes:
    #         print(node)
    #         if isinstance(node, ast.If):
    #             # Parse each if statement and create corresponding nodes
    #             parsed_node = self.parse_if_block(node)
    #             if not root:
    #                 root = parsed_node
    #             else:
    #                 current_node.success = parsed_node
    #             current_node = parsed_node
    #         elif isinstance(node, ast.Expr):
    #             # Handle expressions (function calls like say(), pick(), etc.)
    #             if current_node is None:
    #                 root = LLMTreeNode(ast.unparse(node.value))
    #                 current_node = root
    #             else:
    #                 current_node.success = LLMTreeNode(ast.unparse(node.value))
    #                 current_node = current_node.success

    #     return root

    # def convert_to_task(self, root: LLMTreeNode, parent_operation_name: str = None, success: bool = True):
    #     """Recursively convert the tree into a task by adding operations and connecting them."""
    #     if root is None:
    #         return

    #     # Create the operation
    #     operation_ret = eval("self." + root.function_call)
    #     intermediate_operation_name = None

    #     if isinstance(operation_ret, tuple):
    #         root_operation_name = operation_ret[1]
    #         intermediate_operation_name = operation_ret[0]
    #     else:
    #         root_operation_name = operation_ret

    #     # Connect to parent
    #     if parent_operation_name is not None:
    #         if success:
    #             if intermediate_operation_name is not None:
    #                 # Handle intermediate success connections
    #                 self.task.connect_on_success(parent_operation_name, intermediate_operation_name)
    #                 self.task.connect_on_success(intermediate_operation_name, root_operation_name)
    #             else:
    #                 self.task.connect_on_success(parent_operation_name, root_operation_name)
    #         else:
    #             if intermediate_operation_name is not None:
    #                 # Handle intermediate failure connections
    #                 self.task.connect_on_failure(parent_operation_name, intermediate_operation_name)
    #                 self.task.connect_on_success(intermediate_operation_name, root_operation_name)
    #             else:
    #                 self.task.connect_on_failure(parent_operation_name, root_operation_name)

    #     # Link back to parent on failure if needed
    #     if intermediate_operation_name is not None:
    #         self.task.connect_on_failure(root_operation_name, intermediate_operation_name)
    #         self.task.connect_on_failure(intermediate_operation_name, parent_operation_name)
    #     elif parent_operation_name is not None:
    #         self.task.connect_on_failure(root_operation_name, parent_operation_name)

    #     # Recursively process success and failure branches
    #     if root.success:
    #         self.convert_to_task(root.success, root_operation_name, success=True)
    #     if root.failure:
    #         self.convert_to_task(root.failure, root_operation_name, success=False)


    def compile(self):
        """Compile the LLM plan into a task"""
        
        # self._operation_naming_counter = 0
        # self.task = Task()
        # self.nodes = []

        llm_plan_lines = self.llm_plan.split("\n")

        llm_plan_lines_shifted = [llm_plan_lines[0]]
        llm_plan_lines_shifted.append("    if say('On it!'):")
        for line in llm_plan_lines[1:]:
            llm_plan_lines_shifted.append("    " + line)

        self.llm_plan = "\n".join(llm_plan_lines_shifted)

        self.llm_plan = self.add_self_prefix(self.llm_plan).replace("self.execute", "execute")

        # print("Updated LLM plan:")
        # print(repr(self.llm_plan))

        # print(self.llm_plan)
        exec(self.llm_plan)
        # eval(self.llm_plan)

        execute_task(self.go_to, self.pick, self.place, self.say, self.wave, self.open_cabinet, self.close_cabinet, self.get_detections)

        # tree = ast.parse(self.llm_plan)
        # self.root = self.build_tree(tree.body[0].body)
        # self.convert_to_task(self.root)

        # return self.task
