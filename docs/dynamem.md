# Dynamem

[![arXiv](https://img.shields.io/badge/arXiv-2401.12202-163144.svg?style=for-the-badge)](https://arxiv.org/abs/2411.04999)
![License](https://img.shields.io/github/license/notmahi/bet?color=873a7e&style=for-the-badge)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-262626?style=for-the-badge)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/Videos-Website-db6a4b.svg?style=for-the-badge&logo=airplayvideo)](https://dynamem.github.io)

Dynamem is an open vocabulary mobile manipulation system that works life long in any unseen environments. It can continuously process text queries in the form of "pick up A and place it on B" (e.g. "pick up apple and place it on the plate"). 

Compared to Stretch AI Agent mentioned [here](llm_agent.md), Dynamem can
* continuously update its semantic memory when they observe the environment changes, which allows the system to work life long in homes without rescanning the environments;
* pick up more objects, especially bowl like objects.

However, there is some reason why sometimes we should still use AI agent:
* Dynamem does an open loop pick up, which requires the robot urdf to be very well calibrated as the robot does not take new observation to correct itself once the action plan is generated.
* Dynamem uses Anygrasp, a closed source gripper pose prediction model. Some researchers or companies might not be allowed or be able to use it.

_Click to follow the link to YouTube:_

[![Example of Dynamem in the wild](https://i9.ytimg.com/vi/oBHzOfUdRnE/mqdefault.jpg?sqp=CMD0nboG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGGUgXShTMA8=&rs=AOn4CLAOlWNMyxe1WcShGRpP1BaH3wK2bg)](https://youtu.be/oBHzOfUdRnE)

[Above](https://youtu.be/oBHzOfUdRnE) shows Dynamem running in NYU kitchen.

# Understanding Dynamem code structure
Dynamem consists of three components, navigation, picking, and placing. 
To complete "Pick up A and Place it on B", it will call 4 commands sequentially:
- `navigate(A)`
- `pick(A)`
- `navigate(B)`
- `place(B)`
Besides these commands, Dynamem also provides exploration module
- `explore()`

## Navigation and exploration
Dynamem stores (two) voxelized pointcloud for navigation and exploration. The first pointcloud is used to generate obstacle map for running A* path planning while another is used to store vision language features for visual grounding and generate value map for exploration.

In [Dynamem paper](https://arxiv.org/pdf/2411.04999), three ways to query semantic memory are introduced, for now we only set up querying with vision language feature similarity. We will soon set up querying with the hybrid of mLLMs and vision language feature similarity.

## Picking and placing
Dynamem has two manipulation systems, one is Stretch AI Visual Servoing code, as described in the [LLM agent](llm_agent.md) while another is [OK-Robot manipulation](https://github.com/ok-robot/ok-robot/tree/main/ok-robot-manipulation)

To run [OK-Robot manipulation](https://github.com/ok-robot/ok-robot/tree/main/ok-robot-manipulation), you need to follow [OK-Robot installation instructions](https://github.com/ok-robot/ok-robot/tree/main?tab=readme-ov-file#installation) to prepare AnyGrasp.

# Running Dynamem
You should follow the these instructions to run Dynamem. SLAM and control codes are supposed to be run on the robot while perception models are supposed to be run on the workstation (e.g. a laptop, a lambda machine; might also be run on the robot but not recommended).

## On the robot
### Startup
Once you turn on Stretch robot, you should first calibrate it
```
stretch_free_robot_process.py
stretch_robot_home.py
```
If you have already run these codes to start up the robot, you may move to the next step.

### Launch SLAM on robots

To run navigation system of [Dynamem](https://dynamem.github.io), you first need to install environment with these commands:
```
bash ./install.sh
```

Then we launch SLAM on the robot
```
ros2 launch stretch_ros2_bridge server.launch.py
```
Or you can use docker to launch SLAM
```
docker ./scripts/run_stretch_ai_ros2_bridge.sh
```

For more information, see the [Stretch AI startup guide](start_with_docker_plus_virtenv.md).

## On the workstation

Most of AI codes (e.g. VLMs, mLLMs) should be run on the workstation.

### Specifying IPs in Dynamem scripts

Firstly you should know the ip address of your robot and workstation by running `ifconfig` on these two machines. Continuously tracking ips of different machines is an annoying task. We recommend using [Tailscale](https://tailscale.com) to manage a series of virtual ip addresses. Run following command on the workstation to run dynamem
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP
```
`robot_ip` is used to communicate robot and `server_ip` is used to communicate the server where AnyGrasp runs. If you don't run anygrasp (e.g. navigation only or running Stretch AI visual servoing manipulation instead), then set `server_ip` to `127.0.0.1`.
If you plan to run AnyGrasp on the same workstation, we highly recommend you find the ip of this workstation instead of naivly setting `server_ip` to `127.0.0.1`.

### Loading from previous semantic memory
Dynamem stores the semantic memory as a pickle file after initial rotation in place and everyt time `navigate(A)` is executed. This allows Dynamem to read from saved pickle file so that it can directly load semantic memory from previous runs without rotating in place and scanning surroundings again.

You can control memory saving and reading by specifying `input-path` and `output-path`. 

By specifying `output-path`, the semantic memory will be saved to `specified-output-path` + `.pkl`; otherwise, the semantic memory will be saved to pickle file named by the current datetime in `dynamem_log/`.

By specifying `intput-path`, the robot will first read semantic memory from specified pickle file and will skip the rotating in place.

The command looks like this

```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP --output-path #PICKLE_FILE_PATH --input-path $PICKLE_FILE_PATH
```

### Running with AnyGrasp
TBA. Now you can follow [OK-Robot installation instructions](https://github.com/ok-robot/ok-robot/tree/main?tab=readme-ov-file#installation) to install AnyGrasp and follow [OK-Robot running instructions](https://github.com/ok-robot/ok-robot/tree/main?tab=readme-ov-file#on-workstation) to run AnyGrasp.

### Running without AnyGrasp

If you do not have access to AnyGrasp, you can run with the Stretch AI Visual Servoing code, as described in the [LLM Agent documentation](llm_agent.md). In this case, you can run Dynamem with the following command:
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP --visual-servo
```

### Running with the LLM Agent

You can also run an equivalent of the [LLM agent](llm_agent.md) with Dynamem. In this case, you can run Dynamem with the following command:
```
python -m stretch.app.run_dynamem --use-llm
```

All of the flags [in the agent documentation](llm_agent.md) are also available in Dynamem:
```
# Start with voice chat
python -m stretch.app.run_dynamem --use-llm --use-voice
```

You can specify an LLM, e.g.:
```bash
# Run Gemma 2B from Google locally
python -m stretch.app.run_dynamem --use-llm --llm gemma2b

# Run Openai GPT-4o-mini on the cloud, using an OpenAI API key
OPENAI_API_KEY=your_key_here
python -m stretch.app.run_dynamem --use-llm --llm openai
```

<!-- ### Select between different options to control your system

You will be asked to choose between some options when running.

If you want the robot to explore the environment, select E between E, N, S.
![explore](./images/dynamem_instruction2.jpg)

If you want the robot to run an OVMM task, select N between E, N, S; select y (yes) for the next question; enter text query.
![navigate](./images/dynamem_instruction1.jpg)

After the robot successfully navigates to the target object it will ask you whether you want to pick it up. Since we currently do not support manipulation, just select n (no).

Following this pattern, choose not to place the object and you will be asked to select between E, N, S again.

![no](./images/dynamem_instruction3.jpg)

To quit from the process, just select S. -->

## Cite Dynamem

If you find Dynamem useful in your research, please consider citing:
```
@article{liu2024dynamem,
  title={DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation},
  author={Liu, Peiqi and Guo, Zhanqiu and Warke, Mohit and Chintala, Soumith and Paxton, Chris and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2411.04999},
  year={2024}
}
```
