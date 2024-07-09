# Data Collection for Stretch

## Prerequisites:

### On PC:

- Follow [instructions](../README.md#advanced-installation) for advanced installation of stretch_ai with Python 3.10

  - Advanced installation is only needed if you also want to train/evaluate policies with GPU, pure data collection should be fine with [normal installation](data_collection.md#on-robot)

- [Prepare URDFs and camera calibration](https://github.com/hello-robot/stretch_dex_teleop?tab=readme-ov-file#generate-specialized-urdfs) for dex teleop

- Install our fork of lerobot in a new conda environment:

  ```bash
  conda create -y -n lerobot python=3.10 && conda activate lerobot
  git clone git@github.com:hello-yiche/lerobot.git
  cd lerobot
  git switch stretch-act

  pip3 install -e .
  ```

### On Robot:

- Install normal installation of stretch_ai

  ```bash
    git clone git@github.com:hello-robot/stretch_ai.git
    cd stretch_ai/src
    pip3 install -e .
  ```

- [Prepare URDFs for dex teleop](https://github.com/hello-robot/stretch_dex_teleop?tab=readme-ov-file#generate-specialized-urdfs)

## Quickstart: Record Some Data

On the robot run:

```
python -m stretch.app.dex_teleop.follower
```

On the PC run:

```bash
python -m stretch.app.dex_teleop.leader -i $ROBOT_IP --teleop-mode base_x --save-images --record-success --env-name default
```

You can now record demonstrations by pressing `spacebar` to start and stop recording. See [Recording demonstrations with Dex Teleop](data_collection.md#recording-demonstrations-with-dex-teleop) for more details. After a trial is over, press y/n to record if it was successful.

## Recording demonstrations with Dex Teleop

1. Launch dex teleop follower on robot

   ```bash
   # Launch this command from the directory where URDFs are stored
   python3 -m stretch.app.act.act_follower
   ```

1. Launch dex teleop leader on PC

   Currently supported teleop modes include: `base_x`, `rotary_base`, and `stationary_base`.

   ```bash
   # Launch this command from the directory where URDFs are stored
   # The -s flag enables png images to be saved in addition to videos, which is faster for model training if training is CPU bound (no video decoding)

   python3 -m stretch.app.dex_teleop.leader -i <ip-of-robot> --env-name <name-of-task> --teleop-mode <teleop-mode> -s
   ```

For example

```bash
python3 -m stretch.app.dex_teleop.leader -i $ROBOT_IP --env-name default_task --teleop-mode base_x -s
```

1. Record episodes

   - Press `spacebar` to start recording a demonstration, press `spacebar` again to end demonstration
   - Demonstrations will be stored in stretch_ai/data/default_task/default_user/`name-of-task`

## Format data and push to huggingface repo

1. [Authenticate with huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

1. Process and push demonstration folder to huggingface repo

   ```bash
   # --raw-dir:  where the episodes for this task are stored
   # --local-dir: where a local copy of the final huggingface dataset will be stored, last two layers of local_dir should be in same format as the repo-id
   # --video: If true, dataset will only contain videos and no images
   # --fps: FPS of demonstrations

   python .\lerobot\scripts\push_dataset_to_hub.py \
   --raw-dir /path/to/raw/dir \
   --raw-format dobbe \
   --local-dir ../data/default_task/default_user/hellorobotinc/<your-dataset-name> \
   --video 0 \
   --fps 15 \
   --repo-id hellorobotinc/<your-dataset-name>
   ```

1. Visualizing dataset with Rerun.io

   ```bash
   # Specify root if you wish to use local copy of the dataset, else dataset will be pulled from web
   python .\lerobot\scripts\visualize_dataset.py \
   --repo-id hellorobotinc/<your-dataset-name> \
   --episode-index <episode-idx> \
   --root ../data/default_task/default_user
   ```

## Train a policy

`policy=stretch_diffusion` tells the script to use the configs found in ./lerobot/configs/policy/stretch_diffusion.yaml

`env=stretch_real` indicates that we are using the stretch in a real world env, using configs in ./lerobot/configs/env/stretch_real.yaml

Training configs defined in the policy yaml file can be overridden.
If the config looks like below:

```yaml
training:
  learning_rate: 0.001
```

At runtime we can override this by adding the snippet below. For more details see [Hydra docs](https://hydra.cc/docs/intro/) and [LeRobot](https://github.com/huggingface/lerobot?tab=readme-ov-file#train-your-own-policy).

```bash
training.learning_rate=0.00001
```

Sample training command:

```bash
python3 lerobot/scripts/train.py \
policy=stretch_diffusion \
env=stretch_real \
wandb.enable=true \
training.batch_size=64 \
training.num_workers=16
```

## Evaluating a policy

### On Robot:

```bash
python3 -m stretch.app.act.act_follower
```

### On PC:

Specify the policy of the weights provided:

- Available policies: `diffusion`,`act`

Specify the teleop mode according to the teleop mode used to train the policy

- Available teleop modes: `standard`,`rotary_base`,`stationary_base`,`base_x`,`old_stationary_base`

```bash
python3 -m stretch.app.act.act_leader \
-i <robot-ip> \
--policy_name <name-of-policy> \
--policy_path <path-to-weights-folder> \
--teleop-mode <teleop-mode>
```
