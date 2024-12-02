# Open Vocabulary Mobile Manipulation

Leverage LLMs to generate code policies and VLMs to generate online semantic memory to perform long-horizon tasks. 

## Pipeline

1. **Open Vocabulary Mobile Manipulation (OVMM)**: Use LLMs such as Qwen to generate code policies for mobile manipulation tasks.
2. **Plan Generation**: We use Abstract Syntax Trees (ASTs) to generate plans from the code policies. Each statement is converted into an equivalent managed operation and added to the task queue.
3. **Semantic Memory**: We use vision models such as siglip to generate online semantic memory. This memory is used to perform long-horizon tasks.
4. **Task Execution**: We use the task queue to execute the tasks. The task queue is updated based on the semantic memory.

## Running OVMM

To run OVMM, you first need launch Docker with stretch_ros2_bridge server on your robot with this command:
```
cd stretch_ai && bash scripts/run_stretch_ai_ros2_bridge.sh
```

Then proceed to run OVMM on your PC with this command:
```
cd stretch_ai && bash scripts/run_stretch_ai_gpu_client.sh
python3 -m stretch.app.run_ovmm --robot_ip $ROBOT_IP --enable-realtime-updates
```

- The `robot_ip` is used to communicate with the robot.
- The `enable-realtime-updates` flag is used to enable real-time updates from the robot. The server uses a modified `slam_toolbox` ROS package to send pose graph vertices and edges to the client. The client uses this information to match incoming observations and update the semantic memory.

## Prompt-based Task Execution

After running OVMM, you will be prompted to type in a natural language long-horizon task. The robot will then execute the task using the generated code policies and semantic memory.

### Example Tasks

1. **Pick and Place**: "Pick up the toy and place it inside the box."
2. **Navigation**: "Go to the sofa and pick up the remote."
3. **Engagement**: "Go to the chair and wave at the person and then crack a joke."
