import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

SUPPORTED = ["act", "diffusion"]


def load_policy(policy_name, policy_path, device):

    # Checks if model is supported, then loads model
    policy = None
    if policy_name == "act":
        policy = ACTPolicy.from_pretrained(policy_path)
    elif policy_name == "diffusion":
        policy = DiffusionPolicy.from_pretrained(policy_path)
    else:
        raise NotImplementedError(
            f"{policy_name} is not a supported policy. Supported policies: {SUPPORTED}"
        )
    policy.to(device)
    policy.eval()

    return policy
