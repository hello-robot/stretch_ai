# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from pathlib import Path
from typing import Optional, Tuple

import hydra
import yacs.config
import yaml

import stretch

CONFIG_ROOT = str(Path(stretch.__path__[0]).resolve() / "config")
CONTROL_CONFIG_DIR = str(Path(stretch.__path__[0]).resolve() / "config" / "control")

DATA_ROOT = str(Path(stretch.__path__[0]).parent.resolve() / "../data")


def get_data_path(ext: str) -> str:
    """Returns full path to a particular file in the data directory"""
    return os.path.join(DATA_ROOT, ext)


def get_offload_path(ext: str) -> str:
    """Returns full path to a particular file in the offload directory"""
    folder = os.path.join(DATA_ROOT, "offload", ext)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_scene_path(ext: str) -> str:
    """Returns full path to a particular file in the scene directory"""
    return os.path.join(DATA_ROOT, "scenes", ext)


def get_scene_by_name(name: str) -> str:
    """Returns full path to a particular file in the scene directory"""
    return get_scene_path(f"{name}.xml")


class Config(yacs.config.CfgNode):
    """store a yaml config"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


def get_full_config_path(ext: str) -> str:
    """Returns full path to a particular file"""
    return os.path.join(CONFIG_ROOT, ext)


def get_config(path: str, opts: Optional[list] = None) -> Tuple[Config, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    full_path = get_full_config_path(path)

    # Start with our code's config
    config = Config()
    try:
        config.merge_from_file(full_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path=}, {full_path=}")

    # Add command line arguments
    if opts is not None:
        config.merge_from_list(opts)
    config.freeze()

    # Generate a string representation of our code's config
    config_dict = yaml.load(open(full_path), Loader=yaml.FullLoader)
    if opts is not None:
        for i in range(0, len(opts), 2):
            dict = config_dict
            keys = opts[i].split(".")
            if "TASK_CONFIG" in keys:
                continue
            value = opts[i + 1]
            for key in keys[:-1]:
                dict = dict[key]
            dict[keys[-1]] = value
    config_str = json.dumps(config_dict, indent=4)

    return config, config_str


def load_config(visualize: bool = False, print_images: bool = True, config_path=None, **kwargs):
    """Load config path for real world experiments and use proper presets. This is based on the version from HomeRobot, and is designed therefore to work with Habitat."""
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.VISUALIZE = int(visualize)
    config.PRINT_IMAGES = int(print_images)
    config.EXP_NAME = "debug"
    if config.GROUND_TRUTH_SEMANTICS != 0:
        raise RuntimeError("No ground truth semantics in the real world!")
    config.freeze()
    return config


def get_control_config(cfg_name):
    """Simpler version of the config utility for opening config"""
    with hydra.initialize_config_dir(CONTROL_CONFIG_DIR):
        cfg = hydra.compose(config_name=cfg_name)

    return cfg
