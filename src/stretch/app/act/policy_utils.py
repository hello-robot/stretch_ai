import numpy as np
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torchvision.transforms import v2

import stretch.app.dex_teleop.dex_teleop_parameters as dt

SUPPORTED_POLICIES = ["act", "diffusion"]


def load_policy(
    policy_name: str | None = None, policy_path: str | None = None, device: str | None = "cuda"
):
    """Loads specified policy with name and path. Current supported policies include 'act', 'diffusion'"""
    policy = None
    if policy_name == "act":
        policy = ACTPolicy.from_pretrained(policy_path)
    elif policy_name == "diffusion":
        policy = DiffusionPolicy.from_pretrained(policy_path)
    else:
        raise NotImplementedError(
            f"{policy_name} is not a supported policy. Supported policies: {SUPPORTED_POLICIES}"
        )
    policy.to(device)
    policy.eval()

    return policy


def prepare_state(
    raw_state: dict | None = None, teleop_mode: str | None = None, device: str | None = "cuda"
):
    # Format based on teleop mode
    state = dt.format_state(raw_state, teleop_mode)

    # TODO This mode is only here to support old models with 7 state features. Remove when this is no longer needed
    if teleop_mode == "old_stationary_base":
        state = [
            0.0,  # Placeholder 0 for theta_vel
            state["joint_lift"],
            state["joint_arm_l0"],
            state["joint_wrist_roll"],
            state["joint_wrist_pitch"],
            state["joint_wrist_yaw"],
            state["stretch_gripper"],
        ]
    else:
        # Define explicit order for input state features
        state = [
            state["base_x"],
            state["base_x_vel"],
            state["base_y"],
            state["base_y_vel"],
            state["base_theta"],
            state["base_theta_vel"],
            state["joint_lift"],
            state["joint_arm_l0"],
            state["joint_wrist_pitch"],
            state["joint_wrist_yaw"],
            state["joint_wrist_roll"],
            state["stretch_gripper"],
        ]

    state = torch.from_numpy(np.array(state))
    state = state.to(torch.float32)
    state = state.to(device, non_blocking=True)
    state = state.unsqueeze(0)

    return state


def prepare_image(image, device):

    transforms = v2.Compose([v2.CenterCrop(320)])
    image = torch.from_numpy(image)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)
    image = image.to(device, non_blocking=True)
    image = transforms(image)
    image = image.unsqueeze(0)

    return image


def prepare_observations(
    raw_state: dict | None = None,
    gripper_color_image: np.ndarray | None = None,
    gripper_depth_image: np.ndarray | None = None,
    head_color_image: np.ndarray | None = None,
    head_depth_image: np.ndarray | None = None,
    teleop_mode: str | None = "stationary_base",
    device: str | None = "cuda",
):
    """Prepare state and image observations based on teleop mode and move to specified device"""

    # Prepare state
    state = prepare_state(raw_state, teleop_mode, device)

    # Prepare images
    images = [gripper_color_image, gripper_depth_image, head_color_image, head_depth_image]
    gripper_color_image, gripper_depth_image, head_color_image, head_depth_image = [
        prepare_image(x, device) for x in images
    ]

    observations = {
        "observation.state": state,
        "observation.images.gripper": gripper_color_image,
        "observation.images.head": head_color_image,
        "observation.images.gripper_depth": gripper_depth_image,
        "observation.images.head_depth": head_depth_image,
    }
    return observations


def prepare_action_dict():
    # TODO
    return
