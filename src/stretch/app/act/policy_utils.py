import numpy as np
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torchvision.transforms import v2

SUPPORTED_POLICIES = ["act", "diffusion"]
SUPPORTED_MODES = ["rotary_base", "stationary_base", "base_x"]


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
    state = None
    if teleop_mode == "rotary_base":
        state = [
            raw_state["theta"],
            0.0,  # Placeholder 0 for base_x
            raw_state["joint_lift"],
            raw_state["joint_arm_l0"],
            raw_state["joint_wrist_roll"],
            raw_state["joint_wrist_pitch"],
            raw_state["joint_wrist_yaw"],
            raw_state["stretch_gripper"],
        ]
    elif teleop_mode == "stationary_base":
        state = [
            0.0,  # Placeholder 0 for theta
            0.0,  # Placeholder 0 for base_x
            raw_state["joint_lift"],
            raw_state["joint_arm_l0"],
            raw_state["joint_wrist_roll"],
            raw_state["joint_wrist_pitch"],
            raw_state["joint_wrist_yaw"],
            raw_state["stretch_gripper"],
        ]
    elif teleop_mode == "base_x":
        state = [
            0.0,  # Placeholder 0 for theta
            raw_state["base_x"],
            raw_state["joint_lift"],
            raw_state["joint_arm_l0"],
            raw_state["joint_wrist_roll"],
            raw_state["joint_wrist_pitch"],
            raw_state["joint_wrist_yaw"],
            raw_state["stretch_gripper"],
        ]
    # TODO remove old_stationary_base that only has 7 features (for compatibility with old models)
    elif teleop_mode == "old_stationary_base":
        state = [
            0.0,  # Placeholder 0 for theta
            raw_state["joint_lift"],
            raw_state["joint_arm_l0"],
            raw_state["joint_wrist_roll"],
            raw_state["joint_wrist_pitch"],
            raw_state["joint_wrist_yaw"],
            raw_state["stretch_gripper"],
        ]
    else:
        raise NotImplementedError(
            f"{teleop_mode} is not a supported teleop mode. Supported modes: {SUPPORTED_MODES}"
        )
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
    """Prepares joint state observations according to teleop mode. Current supported modes include 'rotary_base', 'stationary_base', and 'base_x'"""
    state = prepare_state(raw_state, teleop_mode, device)
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
