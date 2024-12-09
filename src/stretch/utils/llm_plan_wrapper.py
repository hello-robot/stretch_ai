# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

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

class LLMPlanWrapper():
    def __init__(self, agent: RobotAgent, llm_plan: str):
        self.agent = agent
        self.robot = agent.robot
        self.llm_plan = llm_plan

    def go_to(self, location: str):
        """Adds a GoToNavOperation to the task"""
        task = Task()
        say_go_to = SpeakOperation(
            name="say_go_to_" + location,
            agent=self.agent,
            robot=self.robot,
        )
        say_go_to.configure(message=f"Going to {location}")
        task.add_operation(say_go_to, True)
        task.run()

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

    def pick(self, object_name: str):
        """Adds a GraspObjectOperation to the task"""
        # Try to expand the frontier and find an object; or just wander around for a while.
        task = Task()
        say_pick = SpeakOperation(
            name="say_pick_" + object_name,
            agent=self.agent,
            robot=self.robot,
        )
        say_pick.configure(message=f"Looking for {object_name}")
        task.add_operation(say_pick, True)
        task.run()

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

        task = Task()
        say_place = SpeakOperation(
            name="say_place_" + receptacle_name,
            agent=self.agent,
            robot=self.robot,
        )
        say_place.configure(message=f"Placing object on {receptacle_name}")
        task.add_operation(say_place, True)
        task.run()

    def say(self, message: str):
        """Adds a SpeakOperation to the task"""
        say_operation = SpeakOperation(
            name="say_" + message + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        say_operation.configure(message=message)
        self.task.add_operation(say_operation, True)
        return "say_" + message + f"_{str(self._operation_naming_counter - 1)}"

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
        task = Task()
        say_wave = SpeakOperation(
            name="say_wave",
            agent=self.agent,
            robot=self.robot,
        )
        say_wave.configure(message="Waving operation not implemented")
        task.add_operation(say_wave, True)
        task.run()

    def open_cabinet(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="open_cabinet" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_not_implemented.configure(message="Open cabinet operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "open_cabinet" + f"_{str(self._operation_naming_counter - 1)}"

    def close_cabinet(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="close_cabinet" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_not_implemented.configure(message="Close cabinet operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "close_cabinet" + f"_{str(self._operation_naming_counter - 1)}"

    def get_detections(self):
        """Adds a SpeakOperation (not implemented) to the task"""
        speak_not_implemented = SpeakOperation(
            name="get_detections" + f"_{str(self._operation_naming_counter)}",
            agent=self.agent,
            robot=self.robot,
        )
        self._operation_naming_counter += 1
        speak_not_implemented.configure(message="Get detections operation not implemented")
        self.task.add_operation(speak_not_implemented, True)
        return "get_detections" + f"_{str(self._operation_naming_counter - 1)}"

    def run(self):
        """Runs the task"""

        # self.task = Task()
        # self._operation_naming_counter = 0

        exec(self.llm_plan)

        execute_task(self.go_to, self.pick, self.place, self.say, self.open_cabinet, self.close_cabinet, self.wave, self.get_detections)

        # return self.task

