import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from torch.nn.functional import conv2d, max_pool2d


def compute_alive_mask(state, alive_threshold):
    return max_pool2d(state[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > alive_threshold


def get_kernels(
    isotropic_only: bool = False, extra_kernels: bool = False, only_vertical: bool = False
):
    # Basic kernels
    identity_kernel = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    average_kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

    # Anisotropic kernels
    sobel_x_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
    sobel_y_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
    sobel_diag = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    prewitt_x = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
    prewitt_y = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3
    scharr_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 16
    scharr_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) / 16
    motion_horizontal = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 0]]) / 3
    motion_vertical = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]]) / 3

    # isotropic kernels
    laplacian_kernel = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]]) / 4
    gaussian_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    log_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    edge_enhance_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    highpass_kernel = identity_kernel - gaussian_kernel

    if isotropic_only and not extra_kernels:
        kernels = [
            identity_kernel,
            average_kernel,
            laplacian_kernel,
            gaussian_kernel,
        ]

    elif isotropic_only and extra_kernels:
        kernels = [
            identity_kernel,
            average_kernel,
            laplacian_kernel,
            gaussian_kernel,
            log_kernel,
            edge_enhance_kernel,
            highpass_kernel,
        ]

    elif not isotropic_only and not extra_kernels:
        kernels = [
            identity_kernel,
            average_kernel,
            laplacian_kernel,
            gaussian_kernel,
            sobel_x_kernel,
            sobel_y_kernel,
            sobel_diag,
        ]

    elif not isotropic_only and only_vertical and not extra_kernels:
        kernels = [
            identity_kernel,
            average_kernel,
            laplacian_kernel,
            gaussian_kernel,
            sobel_y_kernel,
        ]

    elif not isotropic_only and only_vertical and extra_kernels:
        kernels = [
            identity_kernel,
            average_kernel,
            laplacian_kernel,
            gaussian_kernel,
            log_kernel,
            edge_enhance_kernel,
            highpass_kernel,
            sobel_y_kernel,
        ]

    elif not isotropic_only and extra_kernels:
        kernels = [
            # isotropic kernels
            identity_kernel,
            average_kernel,
            laplacian_kernel,
            gaussian_kernel,
            log_kernel,
            edge_enhance_kernel,
            highpass_kernel,
            # anisotropic kernels
            sobel_x_kernel,
            sobel_y_kernel,
            sobel_diag,
            prewitt_x,
            prewitt_y,
            scharr_x,
            scharr_y,
            motion_horizontal,
            motion_vertical,
        ]
    return torch.stack(kernels, dim=0)


def apply_kernels(x, activation=torch.nn.Identity(), isotropic_only=False, extra_kernels=False):

    conv_filters = get_kernels()
    num_kernels = conv_filters.shape[0]

    num_input_channels = x.shape[1]

    conv_weights = conv_filters.unsqueeze(1).repeat(num_input_channels, 1, 1, 1).to(x.device)
    x = activation(conv2d(x, conv_weights, padding=1, groups=num_input_channels))
    num_input_channels *= num_kernels
    return x


def generate_default_kernels(
    num_conv_layers,
    num_output_conv_features,
    num_input_channels,
    isotropic_only,
    extra_kernels,
    kernels=None,
):
    # convolution_mode ="one_kernel_per_channel" or "share_kernels_across_channels"

    if kernels is None:
        kernels = get_kernels(
            isotropic_only=isotropic_only, extra_kernels=extra_kernels
        )  # size K, 3*3 kernel
    elif isinstance(kernels, list):
        kernels = torch.stack(kernels, dim=0)
    # assert kernels.shape[1:] == (F, F), "Kernels must be of shape (K, F, F)", F may be 3 or 5 for instance
    K = kernels.shape[0]  # number of kernels
    # NOTE: weights in pytorch conv 2d should be (Cout, Cin/nGroups, F,F) case so in this convolution mode, Cin*K, 1, 3,3
    all_kernels = []
    for i in range(num_conv_layers):
        if i == 0:  # K, num_input_channels, 3*3
            # For the first convolutional layer, input channels come from num_input_channels
            kernel = kernels.unsqueeze(1).repeat(1, num_input_channels, 1, 1)
        else:
            # For subsequent layers, input channels come from num_output_conv_features
            kernel = kernels.unsqueeze(1).repeat(1, num_output_conv_features, 1, 1)
        assert kernel.shape == (K, num_output_conv_features, 3, 3)
        all_kernels.append(kernel)
    return all_kernels


def seed_python_numpy_torch_cuda(seed: int):
    if seed is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(2**32, size=1))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeeded with {seed}")
    print(f"\nSeeding in pytorch depends on the floating precision used\n")
    return seed


def plot_logger(logger_pd, output_path):
    min_pop_best_eval = logger_pd["pop_best_eval"].min()
    min_mean_eval = logger_pd["mean_eval"].min()

    # # Plot all the losses
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(logger_pd["pop_best_eval"], label="pop_best_eval")
    ax.plot(logger_pd["mean_eval"], label="centroid_eval")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Compliance")
    # wrote min values in legend
    ax.legend([f"pop_best_eval: {min_pop_best_eval:.2f}", f"centroid_eval: {min_mean_eval:.2f}"])
    ax.set_ylim(0, 10000)
    fig.savefig(output_path + "_logger_objective_y_10000lim.png")
    plt.close(fig)

    # # Plot all the losses removing the first 20% of the generations
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # ax.plot(logger_pd["pop_best_eval"][int(0.2 * len(logger_pd)) :], label="pop_best_eval")
    # ax.plot(logger_pd["mean_eval"][int(0.2 * len(logger_pd)) :], label="centroid_eval")
    # ax.set_xlabel("Generation")
    # ax.set_ylabel("Compliance")
    # # wrote min values in legend
    # ax.legend([f"pop_best_eval: {min_pop_best_eval:.2f}", f"centroid_eval: {min_mean_eval:.2f}"])
    # fig.savefig(output_path + "_objective_zoomed.png")
    # plt.close(fig)

    # plot with the y axis limited from 0 to 1000
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(logger_pd["pop_best_eval"], label="pop_best_eval")
    ax.plot(logger_pd["mean_eval"], label="centroid_eval")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Compliance")
    ax.set_ylim(0, 1000)
    ax.legend([f"pop_best_eval: {min_pop_best_eval:.2f}", f"centroid_eval: {min_mean_eval:.2f}"])
    fig.savefig(output_path + "_objective_y_1000lim.png")
    plt.close(fig)

    try:
        # plot mean_density_penalty and std_density_penalty over generation
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(logger_pd["mean_density_penalty"], label="mean_density_penalty")
        ax.plot(logger_pd["std_density_penalty"], label="std_density_penalty")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Density penalty")
        # wrote min values in legend
        min_mean_density_penalty = logger_pd["mean_density_penalty"].min()
        min_std_density_penalty = logger_pd["std_density_penalty"].min()
        ax.legend(
            [
                f"mean_density_penalty: {min_mean_density_penalty:.2f}",
                f"std_density_penalty: {min_std_density_penalty:.2f}",
            ]
        )
        fig.savefig(output_path + "_density_penalty.png")
        plt.close(fig)

        # plot min_density_penalty and min_std_density_penalty over generation
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(logger_pd["density_penalties_min"], label="density_penalties_min")
        ax.plot(logger_pd["std_penalties_min"], label="std_penalties_min")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Density penalty")
        # wrote min values in legend
        min_density_penalties_min = logger_pd["density_penalties_min"].min()
        min_std_penalties_min = logger_pd["std_penalties_min"].min()
        ax.legend(
            [
                f"density_penalties_min: {min_density_penalties_min:.2f}",
                f"std_penalties_min: {min_std_penalties_min:.2f}",
            ]
        )
        fig.savefig(output_path + "_density_penalty_min.png")
        plt.close(fig)

        # plot std_density and mean_density over generation
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(logger_pd["mean_density_abs_distance"], label="mean_density")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Density distance")
        ax.legend()
        fig.savefig(output_path + "_density_distance.png")
        plt.close(fig)

        # plot a figure with objective and density penalty
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(logger_pd["pop_best_eval"][int(0.2 * len(logger_pd)) :], label="pop_best_eval")
        ax.plot(logger_pd["mean_eval"][int(0.2 * len(logger_pd)) :], label="centroid_eval")
        ax.plot(logger_pd["mean_penalties"][int(0.2 * len(logger_pd)) :], label="mean_penalties")
        ax.set_xlabel("Generation")
        ax.legend()
        ax.set_ylim(0, 1000)
        fig.savefig(output_path + "_objective_and_density_penalty.png")
        plt.close(fig)

    except:
        pass


def remove_segment_structure(arr, num_to_remove, threshold, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def is_valid(x, y):
        return 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1] and arr[x, y] > threshold

    def dfs(x, y, count):
        if count == 0:
            return True
        if not is_valid(x, y):
            return False
        arr[x, y] = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        np.random.shuffle(directions)
        for dx, dy in directions:
            if dfs(x + dx, y + dy, count - 1):
                return True
        return False

    ones = np.argwhere(arr > threshold)
    if ones.shape[0] == 0 or ones.shape[0] < num_to_remove:
        print("Not enough elements to remove based on threshold")
        return arr
    start_index = np.random.choice(ones.shape[0])
    start = ones[start_index]
    dfs(start[0], start[1], num_to_remove)

    return arr


def return_activation_function(function_name):
    if function_name == "relu":
        return torch.nn.ReLU()
    elif function_name == "relu6":
        return torch.nn.ReLU6()
    elif function_name == "tanh":
        return torch.nn.Tanh()
    elif function_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif function_name == "identity":
        return torch.nn.Identity()
    elif function_name == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif function_name == "elu":
        return torch.nn.ELU()
    elif function_name == "selu":
        return torch.nn.SELU()
    elif function_name == "softplus":
        return torch.nn.Softplus()
    elif function_name == "clamp(-1,1)":
        return lambda x: torch.clamp(x, -1, 1)
    elif function_name == "clamp(0,1)":
        return lambda x: torch.clamp(x, 0, 1)
    else:
        raise ValueError("Invalid activation function")


def gaussian_field(x, y, zero_low=False):
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, x), torch.linspace(-1, 1, y), indexing="ij")
    if zero_low:
        field = 1 - torch.exp(-(xx**2 + yy**2) * 5)
    else:
        field = 1 - 2 * torch.exp(-(xx**2 + yy**2) * 5)
    return field.unsqueeze(0).unsqueeze(0)


def sinusoidal_fields(x, y, channels, same_direction, zero_low=False):
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, x), torch.linspace(-1, 1, y), indexing="ij")
    freqs = torch.linspace(1, 10, channels)  # Define frequencies for different channels

    a, b = (0.5, 1) if zero_low else (1, 0)

    if same_direction:
        fields = [
            a * (torch.sin(f * xx * 2 * torch.pi) + b) for f in freqs
        ]  # All along the same direction

    else:
        fields = [
            (
                a * (torch.sin(f * xx * 2 * torch.pi) + b)
                if i % 2 == 0
                else a * (torch.cos(f * yy * 2 * torch.pi) + b)
            )
            for i, f in enumerate(freqs)
        ]
    return torch.stack(fields).unsqueeze(0)


def radial_fields(x, y, channels, zero_low=False):
    a, b = (0.5, 1.0) if zero_low else (1, 0)

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, x), torch.linspace(-1, 1, y), indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)  # Compute radial distance
    freqs = torch.linspace(1, 10, channels)

    fields = [a * (torch.sin(f * r * 2 * torch.pi) + b) for f in freqs]

    return torch.stack(fields).unsqueeze(0)


def checkerboard_fields(x, y, channels):
    xx, yy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing="ij")
    fields = [((xx // (2**i) + yy // (2**i)) % 2) * 2 - 1 for i in range(channels)]

    return torch.stack(fields).unsqueeze(0)


def directional_fields(x, y, n, zero_low=False):
    angles = torch.linspace(0, torch.pi, n)
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, x), torch.linspace(-1, 1, y), indexing="ij")
    fields = [
        (
            ((torch.cos(a) * xx + torch.sin(a) * yy) + 1) / 2
            if zero_low
            else (torch.cos(a) * xx + torch.sin(a) * yy)
        )
        for a in angles
    ]
    return torch.stack(fields).unsqueeze(0)


def mix_fields(x, y, n, zero_low=False):
    gaussian = gaussian_field(x, y, zero_low)
    sinusoidal = sinusoidal_fields(x, y, n, True, zero_low)
    directional = directional_fields(x, y, n, zero_low)
    return torch.cat([gaussian, sinusoidal, directional], dim=1)
