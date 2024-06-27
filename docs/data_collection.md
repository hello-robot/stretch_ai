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

## Recording demonstrations with Dex Teleop

1. Launch dex teleop follower on robot

   ```bash
   # Launch this command from the directory where URDFs are stored
   python3 -m stretch.app.act.act_follower
   ```

1. Launch dex teleop leader on PC

   ```bash
   # Launch this command from the directory where URDFs are stored
   # The -s flag enables png images to be saved in addition to videos, which is faster for model training if training is CPU bound (no video decoding)

   python3 -m stretch.app.dex_teleop.leader -i <ip-of-robot> --env-name <name-of-task> -s
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
