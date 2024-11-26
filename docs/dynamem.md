# `Dynamem`

[![arXiv](https://img.shields.io/badge/arXiv-2401.12202-163144.svg?style=for-the-badge)](https://arxiv.org/abs/2411.04999)
![License](https://img.shields.io/github/license/notmahi/bet?color=873a7e&style=for-the-badge)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-262626?style=for-the-badge)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/Videos-Website-db6a4b.svg?style=for-the-badge&logo=airplayvideo)](https://dynamem.github.io)

## How to Run Dynamem

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

### Arguments in Dynamem scripts

On the workstation, we run following command on the workstation to run dynamem
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP [--input-path $PICKLE_FILE_PATH]
```
`robot_ip` is used to communicate robot and `server_ip` is used to communicate the server where AnyGrasp runs (If you only want to try navigation, then set this to `127.0.0.1`)

If you want to run Dynamem directly based on previous runs, you can always find the pickle file saved in previous runs and set `input-path`. If you want to run from scratch, then simply not include `input-path` argument and the robot will first rotate in place to scan surroundings.

#### Running without AnyGrasp

If you do not have access to AnyGrasp, you can run with the Stretch AI Visual Servoing code, as described in the [LLM Agent documentation](llm_agent.md). In this case, you can run Dynamem with the following command:
```
python -m stretch.app.run_dynamem --robot_ip $ROBOT_IP --server_ip $WORKSTATION_SERVER_IP --visual-servo
```

#### Running with the LLM Agent

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

### Select between different options to control your system

You will be asked to choose between some options when running.

If you want the robot to explore the environment, select E between E, N, S.
![explore](./images/dynamem_instruction2.jpg)

If you want the robot to run an OVMM task, select N between E, N, S; select y (yes) for the next question; enter text query.
![navigate](./images/dynamem_instruction1.jpg)

After the robot successfully navigates to the target object it will ask you whether you want to pick it up. Since we currently do not support manipulation, just select n (no).

Following this pattern, choose not to place the object and you will be asked to select between E, N, S again.

![no](./images/dynamem_instruction3.jpg)

To quit from the process, just select S.

### Cite Dynamem

If you find Dynamem useful in your research, please consider citing:
```
@article{liu2024dynamem,
  title={DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation},
  author={Liu, Peiqi and Guo, Zhanqiu and Warke, Mohit and Chintala, Soumith and Paxton, Chris and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2411.04999},
  year={2024}
}
```
