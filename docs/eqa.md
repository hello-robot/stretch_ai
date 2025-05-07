# The Stretch AI EQA Module

The **Embodied Question Answering (EQA) Module** enables a robot to actively explore its environment, gather visual and spatial data, and answer user queries about what it sees. To answer queries, the EQA module has the robot explore the environment to acquire useful information to answer the question, produces a semantic representation of the environment, and processes questions from the user. Systems like the EQA module have the potential to be used in a variety of applications. For example, it might help people find objects in their home, which could be useful to many people, including people with visual and cognitive impairments. It might also help people monitor their home when they're away by enabling them to ask the robot to check on things.

## Demo Video

[The following](https://www.youtube.com/watch?v=MZq1BcG9stQ) shows Stretch AI EQA running in one of our developers' homes.

_Click this large image to follow the link to YouTube:_

[![A demonstration of the EQA module in action](images/eqa.png)](https://www.youtube.com/watch?v=MZq1BcG9stQ)

# Motivation and Methodology

In previous EQA work [GraphEQA](https://arxiv.org/abs/2412.14480), researchers provided a multimodal large language models (mLLMs), such as Google's Gemini and OpenAI's GPT, with a prompt that includes a object-centric semantic scene graph and task-relevant robot image observations. GraphEQA utilizes third party scene graph modules [Hydra](https://arxiv.org/abs/2201.13360) for ROS Noetic. Installing this module can be difficult due to OS and software version compatibility. To provide a more user friendly alternative, we adapted the methods of [GraphEQA](https://arxiv.org/abs/2412.14480) for use with existing code in the Stretch AI repo.

In GraphEQA, mLLMs are expected to answer the question based on task-relevant image observations and plan exploration based on a scene graph string. For the Stretch AI EQA Module, [DynaMem system](dynamem.md) finds task-relevant images and VLM models, such as [Qwen](../src/stretch/llms/qwen_client.py) and [OpenAI GPT](../src/stretch/llms/openai_client.py), extract visual clues from each image observations by listing featured objects in the images such as beds, tables, etc. 

Following this idea, we have designed an EQA pipeline requiring only `a Stretch robot`, `a GPU machine with 12GB VRAM`, and `Internet connection`. Following models (including Cloud API) will be called:
- A light weighted VLM running on the local worstation. We use `Qwen-VL-2.5-3B` here.
- A vision language encoder trained in contrastive manner. We use `SigLip-v1-so400m` here
- A powerful mLLM. We use `gemini-2.5-pro-preview-03-25` here.

When receiving a new question, we will follow this recipe to find out the answer:
- Extract few keywords from the question using the light weighted VLM. For example, `is there a hand sanitizer near sink?` will result in keywords `hand sanitizer` and `sink`.
- Rotate the head pan to look around, follow DynaMem pipeline to add images into voxel-based semantic memory (extract pixel level vision language features with vision language encoder, then project 2D pixels into 3D points to add to the voxel map).
- Use the light weighted VLM to identify featured object names from imagte observation and add these visual clues into a list.
- Query DynaMem to identify few task relevant images.
- Identify image observations selected as task relevant images and image observations corresponding to unexplored frontiers. Add this information to augment visual clues.
- Prompt mLLM with relevant images along with augmented visual clues to answer questions. Following GraphEQA, we also ask mLLM to provide confidence with the answers. If mLLM is not confident with the answers, it should also output an image id indicating areas that should be explored. 
- If no certain answer can be provided, the robot should navigate to the selected image id.
- Iterate the above process until a certain answer can be provided.


## Understanding EQA's code structure

This module shares or extends core dependencies (mapping, perception, llms) with other Stretch AI modules like AI Pickup and DynaMem. Following codes are relevant to this module: 

| File locations                  | Purpose                                                     |
| ----------------------- | ---------------------------------------------------------------- |
| [`src/stretch/app/run_eqa.py`](../src/stretch/app/run_eqa.py)       |       Entry point for EQA module                       |
| [`src/stretch/agent/task/dynamem/dynamem_task.py`](../src/stretch/agent/task/dynamem/dynamem_task.py?plain=1#L409)  | An executor wrapper for EQA module |
| [`src/stretch/agent/robot_agent_eqa.py`](../src/stretch/agent/robot_agent_eqa.py)             | Robot agent class containing all useful APIs for question answering  |
| [`src/stretch/mapping/voxel/voxel_eqa.py`](../src/stretch/mapping/voxel/voxel_eqa.py)         | Robot mapping utilities class extending from [DynaMem voxel.py](../src/stretch/mapping/voxel/voxel_dynamem.py)            |
| [`src/stretch/mapping/voxel/voxel_map_eqa.py`](../src/stretch/mapping/voxel/voxel_map_eqa.py)         | Implement `query answer` functions based on semantic memory  extending from [DynaMem voxel_map.py](../src/stretch/mapping/voxel/voxel_map_dynamem.py) |

## Instructions

### Installation and preparation

The very first step is to install all necessary packages on both your Stretch robot and your workstation following [this instruction](./install_details.md). 

Next you should install Gemini following [Google's docs](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) and obtain a Google API key with Tier 1. Tier 1 belong to Pay-as-you-go catogories. 

**So BE VERY CAUTIOUS! That means you will be charged as you use Gemini models as you attempt EQA module!**.  

But Gemini model usage in this module is fairly cheap. You can check [pricing](https://ai.google.dev/gemini-api/docs/pricing) and [rate limit](https://ai.google.dev/gemini-api/docs/rate-limits) for `gemini-2.5-pro-preview-03-25`.

If you also want to try Discord bot, which is a more beautiful, user friendly communication interface compared with the naive terminal and command line, you should also need to install dependencies and obtain your discord tokens following [discord_bot.md](./discord_bot.md)

### Run EQA module

Launch the EQA agent via the `run_eqa` entry-point. By default, the robot will first rotate in place to scan its surroundings, pop out a rerun window (but the rerun contents will not be automatically saved, once you close the rerun window, you lose all visualization data), and you will be asked to enter your questions in the terminal.

You need to know the ip address of your robot to send commands to your robot. Once you know your `ROBOT_IP`, you can start running the following commands to try this EQA module. 

You also need to set up your Gemini key before running EQA scripts by

```bash
export GOOGLE_API_KEY=$YOUR_GEMINI_TOKEN
```

If you also want to try discord bot, you need to set up discord token as well

```bash
export DISCORD_TOKEN=$YOUR_DICORD_TOKEN
```

```bash
python -m stretch.app.run_eqa --robot_ip $ROBOT_IP
```

Other options

- `--not_rotate_in_place`, `-N` : skip initial rotation-in-place scan
- `--discord`, `-D`: launch Discord bot for a better interface than the terminal and command line
- `--save_rerun`, `--SR`: save Rerun log files to `dynamem_log/debug_*` as rrd file for offline replay (but rerun window online streaming would be disabled)

**Example runs**:
Assume your robot ip is `192.168.1.42`.
* Skip initial rotation-in-place scan:

  ```bash
  python -m stretch.app.run_eqa --robot_ip 192.168.1.42 -N
  ```
* Enable Discord for remote users:

  ```bash
  python -m stretch.app.run_eqa --robot_ip 192.168.1.42 -D
  ```
* No initial rotation-in-place scan, save rerun visualization, enable discord:

  ```bash
  python -m stretch.app.run_eqa --robot_ip 192.168.1.42 -N -D --SR
  ```


## Contributing

This is an active component within the Stretch repository. Please follow the main [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines for branching, testing, and pull requests.

---

*Last updated: May 2025*
