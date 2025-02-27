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

[![Example of Dynamem in the wild](images/dynamem.png)](https://youtu.be/oBHzOfUdRnE)

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

In [Dynamem paper](https://arxiv.org/pdf/2411.04999), three ways to query semantic memory for visual grounding are introduced, in this stack we only set up querying with vision language feature similarity and querying with the hybrid of mLLMs and vision language feature similarity. The first strategy is faster while the second has better performance. By default the stack will chose VL feature similarity to do visual grounding.

In terms of exploration, we discovered that commonly used frontier based exploration (FBE) is not suitable for dynamic environments because obtacles might be moved around, creating new frontier, and already scanned portions of the room might also be changed. Therefore, we introduced a value based exploration that assigns any point in the 2D map a heuristic value evaluating how valuable it is to explore to this point. The detailed analysis is described in [Dynamem paper](https://arxiv.org/pdf/2411.04999).

## Picking and placing
Dynamem has two manipulation systems, one is Stretch AI Visual Servoing code, as described in the [LLM agent](llm_agent.md) while another is [OK-Robot manipulation](https://github.com/ok-robot/ok-robot/tree/main/ok-robot-manipulation).

Instructions for AnyGrasp manipulation is put [here](#manipulation-with-anygrasp) and instructions for visualn servoing manipulation is put [here](#manipulation-with-stretch-ai-visual-servoing-manipulation).

The high level idea for AnyGrasp picking is
- Transform RGBD image from Stretch head camera into a RGB pointcloud.
- [AnyGrasp](https://arxiv.org/abs/2212.08333) proposes a set of collision free gripper poses given a RGB pointcloud.
- [OWLv2](https://arxiv.org/abs/2306.09683) and [SAMv2](https://ai.meta.com/blog/segment-anything-2/) to select only gripper poses that actually manipulates the target object.
- Transform the selected 6-DoF pose into gripper actions using URDF.

Placing is relatively simpler as all you need to do is to segment the target receptacle in the image and select a middle point to drop on.

The advantages of AnyGrasp manipulation system, compared to visual servoing manipulation in [LLM agent](llm_agent.md) includes:
- More general purpose, dealing with objects with different shapes, such as bowls, bananas.
The disadvantages includes:
- Open loop so unable to recover from controller errors.
- Reliance on accurate robot calibration and urdf.

# Running Dynamem
You should follow the these instructions to run Dynamem. SLAM and control codes are supposed to be run on the robot while perception models are supposed to be run on the workstation (e.g. a laptop, a lambda machine; might also be run on the robot but not recommended).

So you should clone stretch ai repo with this command
```
git clone https://github.com/hello-robot/stretch_ai.git --recursive
cd stretch_ai
```
On **BOTH** your robot and workstation.

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

Next you are going to set up your robot launch files, please follow instructions in [Stretch AI startup guide](start_with_docker_plus_virtenv.md) to set up either [Docker](start_with_docker_plus_virtenv.md#run-the-robots-script) or [ROS2](start_with_docker_plus_virtenv.md#installing-ros2-packages-without-docker).

Then we launch SLAM on the robot.

If you choose to install with ROS2, run
```
ros2 launch stretch_ros2_bridge server.launch.py
```
Or if you choose to use docker, run
```
bash ./scripts/run_stretch_ai_ros2_bridge_server.sh --update
```

For more information on how to launch your robot, see the [Stretch AI startup guide](start_with_docker_plus_virtenv.md).

## On the workstation

Most of AI codes (e.g. VLMs, mLLMs) should be run on the workstation. 

You need to first install the conda environments on the workstation, we recommend you run
```
./install.sh --no-version
mamba activate stretch_ai
```
If you use visual servo manipulation, you would need to further install SAM2
```
cd third_party/segment-anything-2
pip install -e .
```
If you use AnyGrasp manipulation, please refer to [these instructions](#prepare-manipulation-with-anygrasp) for the installation, 
you would need to create a new conda environment on your worstation.

### Copying URDF from robot to workstation
No matter whether you choose to run which manipulation, having a well calibrated robot URDF is important, you should follow these steps to set up robot URDF (while visual servo picking does not require accurate robot URDF, placing heuristic is shared between these two systems):
- On your robot, follow instructions described in [Stretch Ros2](https://github.com/hello-robot/stretch_ros2/tree/humble/stretch_calibration) to calibrate your robot.
- Once you have a well calibrated urdf (in `~/ament_ws/src/stretch_ros2/stretch_description/urdf/stretch.urdf` on your stretch robot), copy it to your workstation `src/stretch/config/urdf/stretch.urdf`. It is recommended to run following commands on your workstation:
```
scp hello-robot@[ROBOT IP]:~/ament_ws/src/stretch_ros2/stretch_description/urdf/stretch.urdf stretch_ai/src/stretch/config/urdf/
```
- Run the following python scripts to replace urdf modification described in [OK Robot calibration docs](https://github.com/ok-robot/ok-robot/blob/main/docs/robot-calibration.md) 
```
python src/stretch/config/dynamem_urdf.py --urdf-path src/stretch/config/urdf/stretch.urdf
```

Note that while URDF calibration is important for both manipulation systems, AnyGrasp manipulation has much higher requirement on robot calibration. On the other hand, even though the calibration is not perfect in visual servo manipulation, in most cases the robot is still going to complete the task.

You might want to check your calibration if the following things happen:
- Floor in the navigation pointcloud does not fall on `z=0` plane.
- Manipulation does not follow AnyGrasp predictions.

### Specifying IPs in Dynamem scripts

Firstly you should know the ip address of your robot and workstation by running `ifconfig` on these two machines. Continuously tracking ips of different machines is an annoying task. We recommend using [Tailscale](https://tailscale.com) to manage a series of virtual ip addresses. Run following command on the workstation to run dynamem
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP -S
```
`robot_ip` is used to communicate robot and `server_ip` is used to communicate the server where AnyGrasp runs. If you don't run anygrasp (e.g. navigation only or running Stretch AI visual servoing manipulation instead), then set `server_ip` to `127.0.0.1` or just leave it blank.
If you plan to run AnyGrasp on the same workstation, we highly recommend you find the ip of this workstation instead of naivly setting `server_ip` to `127.0.0.1`.

Once the robot starts doing OVMM, a rerun window will be popped up to visualize robot's thoughts.
![Example of Dynamem in the wild](images/dynamem_rerun.png)

### Manipulation with AnyGrasp
The very first thing is to make sure OK-Robot repo is a submodule in your Stretch AI repo in `third_party/`!!! 
If not, run `git submodule update --init --recursive` to update all submodules.

Next, please strictly follow [aforementioned steps](#copying-urdf-from-robot-to-workstation) to prepare accurate robot URDF!!!

Few steps are needed to be done before you can try AnyGrasp:
- Since AnyGrasp is a closed source model, you should first request for AnyGrasp license following [These instructions](https://github.com/graspnet/anygrasp_sdk?tab=readme-ov-file#license-registration)
- Install a new conda environment for running anygrasp following [OK Robot environment installation instructions](https://github.com/ok-robot/ok-robot/blob/main/docs/workspace-installation.md). **NOTE** that `stretch_ai` environment does not support AnyGrasp because the AnyGrasp packages conflict with `stretch_ai`'s python version.
- Run AnyGrasp with following commands in a new terminal window
```
# If you have not yet activated anygrasp conda environment, do so.
conda activate ok-robot-env

# Assume you are in stretch_ai folder in the new window.
cd third_party/ok-robot/ok-robot-manipulation/src/
python demo.py --open_communication --port 5557
```
To understand more options in running AnyGrasp, please read [OK Robot Manipulation](https://github.com/ok-robot/ok-robot/tree/main/ok-robot-manipulation).

After AnyGrasp is launched, you can run default Dynamem commands as described above.
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP
```

### Two DynaMem modes: exploration and manipulation
Dynamem support both exploration & mapping and OVMM tasks. So before each task it will ask you whether you want to run E (denoted for exploration) and M (denoted for OVMM).

One exploration iteration includes
* Looking around and scanning new RGBD images;
* Moving towards the point of interest in the map.
To specify how many exploration iterations you want the robot to run after selecting exploration, set up `explore-iter`. For example, if you want the robot to explore for 5 iterations, use the command.
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP -S --explore-iter 5
```

### Visual grounding with GPT4o
[As mentioned previously](#navigation-and-exploration), by default we run visual grounding by doing object detection on the robot observataion with the highest cosine similarity. While this strategy is fast, another querying strategy, prompting GPT-4o to process top-k robot observations has better accuracy. 

To try this querying strategy that uses GPT-4o boost your navigation accuracy, you first need to follow [OPENAI's instructions](https://platform.openai.com/docs/overview) to create API keys. After that you can try this version by turning on mllm(`-M`) in your scripts:
```
OPNEAI_API_KEY=$YOUR_API_KEY python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP -S -M
```

### Loading from previous semantic memory
Dynamem stores the semantic memory as a pickle file after initial rotation-in-place and every time `navigate(A)` is executed. This allows Dynamem to read from saved pickle file so that it can directly load the semantic memory from previous runs without rotating in place and scanning surroundings again.

You can control memory saving and reading by specifying `input-path` and `output-path`. 

By specifying `output-path`, the semantic memory will be saved to `dynamem_log/` + `specified-output-path` + `.pkl`; otherwise, the semantic memory will be saved to pickle file named by the current datetime in `dynamem_log/`.

By specifying `intput-path`, the robot will first read semantic memory from specified pickle file and will skip the rotating in place.

The command looks like this

```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP --output-path $PICKLE_FILE_PATH --input-path $PICKLE_FILE_PATH -S
```

### Ask for humans' confirmations before doing each subtask
Dynamem OVMM task implementation hardcodes such API calling sequence: navigating to the target object `navigate(A)`, picking up the object `pick(A)`, navigating to the target receptacle `navigate(B)`, placing the object on the receptacle `place(B)`. However, sometimes we might want to interfere with robot task planning. For example, if first picking up fails, we humans might want the robot to try again. 

So how can we steer robot actions? One functionality we provided is asking for humans' confirmations. That is to say, even though by default the system still calls `navigate(A)`, `pick(A)`, `navigate(B)`, `place(B)` in sequence, but before it implements each module, humans can explicitly tell the robot whether they want it to call this API call. 

How is that functionality helpful? Sometimes when the robot is already facing the object, we might not want to waste time in navigation, by selecting `N` (no) when asked "Do you want to run navigation?", the robot can skip navigation and directly pick up objects.

The flag `-S` in previous commands, it configures Dyname to skip these human confirmantions. To enable this functionality, you need to run

```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP
```

### Manipulation with Stretch AI visual servoing manipulation

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
