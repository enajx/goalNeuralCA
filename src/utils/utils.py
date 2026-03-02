import torch
import os
from tqdm import tqdm
import numpy as np
import time
import random
import os
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import PIL.Image
import requests
import yaml
import pathlib
from torch.nn import functional as F


def return_activatin_fn(activation_fn_str, prelu_num_channels=None):

    if activation_fn_str is None:
        return None

    activation_fn_str = activation_fn_str.lower()
    if activation_fn_str.replace(" ", "") == "clamp(-1,1)":
        return lambda x: torch.clamp(x, -1, 1)
    elif activation_fn_str.replace(" ", "") == "clamp(0,1)":
        return lambda x: torch.clamp(x, 0, 1)
    elif activation_fn_str == "tanh":
        return torch.nn.Tanh()
    elif activation_fn_str == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation_fn_str == "relu":
        return torch.nn.ReLU()
    elif activation_fn_str == "relu6":
        return torch.nn.ReLU6()
    elif activation_fn_str == "identity":
        return torch.nn.Identity()
    elif activation_fn_str == "leakyrelu" or activation_fn_str == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif activation_fn_str == "softplus":
        return torch.nn.Softplus()
    elif activation_fn_str == "prelu":
        if prelu_num_channels is None:
            return torch.nn.PReLU()
        else:
            return torch.nn.PReLU(prelu_num_channels)
    else:
        raise ValueError("Invalid state_activation_hidden")


def seed_python_numpy_torch_cuda(seed: int):
    if seed is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(2**32, size=1)[0])
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeeded with {seed}")
    return seed


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def get_activation_functions(
    config, num_kernels, nca_channels, hidden_dim_mlp, external_embedding_dim
):
    if (
        "prelu" in config["activation_conv"]
        or "prelu" in config["activation_fc"]
        or "prelu" in config["activation_last"]
    ):
        prelu_per_channel = config["prelu_per_channel"]
        prelu_channels_conv = num_kernels * nca_channels if prelu_per_channel else None
        prelu_channels_fc = (
            hidden_dim_mlp[-1]
            if prelu_per_channel and hidden_dim_mlp
            else num_kernels * nca_channels + external_embedding_dim if prelu_per_channel else None
        )
        prelu_channels_last = (
            hidden_dim_mlp[-1]
            if prelu_per_channel and hidden_dim_mlp
            else prelu_channels_fc if prelu_per_channel else None
        )
    else:
        prelu_channels_conv = None
        prelu_channels_fc = None
        prelu_channels_last = None

    activation_conv = return_activatin_fn(config["activation_conv"], prelu_channels_conv)
    activation_fc = return_activatin_fn(config["activation_fc"], prelu_channels_fc)
    activation_last = return_activatin_fn(config["activation_last"], prelu_channels_last)

    return activation_conv, activation_fc, activation_last


def load_model_from_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_model_config_yaml(path, model, config):
    config_ = {
        "num_input_channels": model.num_input_channels,
        "num_external_channels": model.num_external_channels,
        "num_output_conv_features": (
            None
            if model.convolution_mode == "share_kernels_across_channels"
            else model.num_output_conv_features
        ),
        "num_conv_layers": model.num_conv_layers,
        "hidden_dim_mlp": model.hidden_dim_mlp,
        "bias": model.bias,
        "activation_conv": model.activation_conv.__class__.__name__,
        "activation_fc": model.activation_fc.__class__.__name__,
        "activation_last": (
            model.activation_last.__class__.__name__
            if isinstance(model.activation_last, torch.nn.Module)
            else "custom"
        ),
        "stochastic_update_ratio": model.stochastic_update_ratio,
        "convolution_mode": model.convolution_mode,
        "fixed_kernels": model.fixed_kernels,
        "num_kernels": model.num_kernels,
        "custom_kernels": None,
        "width_kernel": model.width_kernel,
        "additive_update": model.additive_update,
        "merge_ext": config["merge_ext"],
        "alive_mask_goal": config["alive_mask_goal"],
        "alive_threshold": config["alive_threshold"],
        "boundary_condition": config["boundary_condition"],
        "isotropic_only": config["isotropic_only"],
        "extra_kernels": config["extra_kernels"],
    }
    with open(path, "w") as f:
        yaml.dump(config_, f)


def broadcast_external_inputs(goal_encoders_batch, size):
    """
    Broadcast goal encoders to spatial dimensions.

    Args:
        goal_encoders_batch: [batch_size, goal_channels]
        size: spatial size - int for square, or tuple (width, height) for non-square

    Returns:
        [batch_size, goal_channels, H, W]
    """
    if isinstance(size, tuple):
        W, H = size  # (width, height)
    else:
        H, W = size, size

    return goal_encoders_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
