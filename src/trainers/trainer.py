import os
import sys
import time
import shutil

import random

sys.path.append(".")
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
import sys
import wandb
import yaml
import matplotlib.pyplot as plt
import pathlib


from NCAs.NCA_mlp import NCA_mlp, evaluate_nca
from NCAs.utils import (
    gaussian_field,
    directional_fields,
    sinusoidal_fields,
    radial_fields,
    mix_fields,
    return_activation_function,
    compute_alive_mask,
)
from src.utils.utils import broadcast_external_inputs, save_model_config_yaml
from src.visualisation.viz import animate_states, animate_hidden_channels
from NCAs.visualisation_functions import (
    visualize_all_patterns_grid,
    create_animation_grid,
    create_transform_animation,
)
from src.datasets.pattern_dataset import (
    GoalPatternsDataset,
    GoalPatternsTransformDataset,
    GoalPatternsMorphingDataset,
    GoalPatternsTrajectoryDataset,
)
from src.utils.utils_plotting import (
    plot_training_curve,
    plot_checksum_l1_curve,
    plot_morphing_grid,
    plot_transformations_grid,
    plot_gradient_magnitudes,
)

torch.set_printoptions(precision=4)
warnings.filterwarnings("ignore")


def prepare_external_inputs(task_encoders_processed, batch_size, spatial_size):
    """
    Broadcast task encoders to spatial dimensions for a given batch size and spatial size.
    """
    if task_encoders_processed is not None:
        return broadcast_external_inputs(task_encoders_processed, spatial_size)
    else:
        return None


def nca_step(state, model, x_ext, update_noise, additive_update, alive_mask, state_norm):
    if alive_mask:
        pre_life_mask = compute_alive_mask(state, model.alive_threshold)

    if additive_update:
        state = state + model(state, x_ext=x_ext, update_noise=update_noise)
    else:
        state = model(state, x_ext=x_ext, update_noise=update_noise)

    if alive_mask:
        post_life_mask = compute_alive_mask(state, model.alive_threshold)
        life_mask = (pre_life_mask & post_life_mask).float()
        state = state * life_mask

    if state_norm:
        state = torch.clamp(state, 0, 1)

    return state


def evaluate_nca_batched(
    model,
    initial_state,
    task_encoders_processed,
    nca_steps,
    additive,
    state_norm,
    alive_mask,
    update_noise=0.0,
):
    state = initial_state.clone().detach().requires_grad_(False)
    batch_size, _, H, W = state.shape

    combined_external = prepare_external_inputs(task_encoders_processed, batch_size, H)

    states = []
    states.append(state)
    for step in range(nca_steps):
        state = nca_step(state, model, combined_external, update_noise, additive, alive_mask, state_norm)
        states.append(state)

    states = torch.stack(states, axis=0)
    return states[-1], states


def evaluate_checksum_l1_distance(
    config,
    model,
    dataset_to_use,
    external_encoder_layer,
    activation_state_norm,
    device,
):
    """
    Helper function to evaluate checksum L1 distance for patterns_translation task.

    Returns:
        float: Mean checksum L1 distance
    """
    if config["task"] != "patterns_translation":
        return None

    patterns_ = []
    task_encoders_ = []

    for idx in range(len(dataset_to_use)):
        input_img, task_enc, _ = dataset_to_use[idx]  # We don't need the target for checksum eval
        patterns_.append(input_img)
        task_encoders_.append(task_enc)

    patterns_ = torch.stack(patterns_, dim=0).to(device)
    task_encoders_ = torch.stack(task_encoders_, dim=0).to(device)

    # Run checksum evaluation
    with torch.no_grad():
        # Get current model state (not best model)
        current_model = model
        current_model.eval()

        # Transform task encoders through external encoder layer if it exists
        if external_encoder_layer is not None:
            task_encoders_ = external_encoder_layer(task_encoders_)

        # Calculate initial checksums
        initial_checksums = patterns_.sum(dim=(1, 2, 3)).to(device)  # [batch_size]

        final_step_checksum, _ = evaluate_nca_batched(
            current_model.to(device),
            patterns_.to(device),
            task_encoders_.to(device),
            config["steps_checksum_eval"] * config["nca_steps"],  # Use longer evaluation steps
            additive=config["additive_update"],
            state_norm=activation_state_norm,
            alive_mask=config["alive_mask"],
            update_noise=0.0,
        )

        # Calculate final checksums
        final_checksums = final_step_checksum.sum(dim=(1, 2, 3))  # [batch_size]

        # Calculate L1 distances for each sample
        checksum_l1_distances = torch.abs(final_checksums - initial_checksums.to(device))

        # Calculate mean for this evaluation
        mean_checksum_l1 = checksum_l1_distances.mean().item()

        return mean_checksum_l1


def train(config, device, rank, local_rank, distributed=False):
    print(f"\nLaunching training on device {device} with rank {rank} & local rank {local_rank}")

    # Convert dtype string to torch dtype
    if config["dtype"] == "float64":
        dtype = torch.float64
    elif config["dtype"] == "float32":
        dtype = torch.float32
    elif config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    torch.set_default_dtype(dtype)
    if rank == 0:
        print(f"Using default precision with dtype {dtype}")

    # Ensure alive_threshold is always a float
    if isinstance(config["alive_threshold"], str):
        alive_threshold = float(config["alive_threshold"])
    else:
        alive_threshold = config["alive_threshold"]

    # Use all parameters from config
    pattern_size = config["pattern_size"]
    space_size = config["space_size"]
    extra_channels = config["extra_channels"]
    nca_steps = config["nca_steps"]
    embedding_dim = config["embedding_dim"]
    alive_mask_goal = config["alive_mask_goal"]
    alive_mask = config["alive_mask"]

    # Handle nca_steps as tuple for random sampling
    if isinstance(nca_steps, (tuple, list)) and len(nca_steps) == 2:
        nca_steps_min, nca_steps_max = nca_steps
        nca_steps_is_range = True
        # For evaluation, use the middle value as consistent choice
        nca_steps_eval = (nca_steps_min + nca_steps_max) // 2
        if rank == 0:
            print(f"NCA steps will be randomly sampled from range [{nca_steps_min}, {nca_steps_max}] during training")
            print(f"NCA steps for evaluation will be: {nca_steps_eval}")
    else:
        nca_steps_is_range = False
        nca_steps_eval = nca_steps

    loss_all_dev_min_t = config.get("loss_all_dev_min_t", None)
    if loss_all_dev_min_t and config["task"] == "patterns_conditional_growth":
        min_nca_steps = nca_steps_min if nca_steps_is_range else nca_steps
        if loss_all_dev_min_t >= min_nca_steps:
            raise ValueError(f"loss_all_dev_min_t ({loss_all_dev_min_t}) must be smaller than the minimum nca_steps ({min_nca_steps})")
        if rank == 0:
            print(f"NCA steps fixed at: {nca_steps}")

    # Get target_patterns parameter from config
    target_patterns = config["target_patterns"]

    # Determine number of patterns
    import os

    if isinstance(target_patterns, str) and target_patterns.endswith("/"):
        num_patterns = len([f for f in os.listdir(target_patterns) if f.endswith(".png")])
    elif isinstance(target_patterns, list):
        num_patterns = len(target_patterns)
    else:
        raise ValueError("target_patterns must be a folder path (ending in '/') or a list")

    # Task-specific parameters
    if config["task"] == "patterns_morphing":
        use_one_hot_encoder = config["use_one_hot_encoder"]
        external_encoder_dim = config["external_encoder_dim"]
        seed_type = config["seed_type"]
        one_hot_dim = num_patterns
    elif config["task"] == "patterns_rotation":
        external_encoder_dim = config["external_encoder_dim"]
        one_hot_dim = 2
    elif config["task"] == "patterns_translation":
        external_encoder_dim = config["external_encoder_dim"]
        one_hot_dim = 4
    elif config["task"] == "patterns_translation_trajectory":
        external_encoder_dim = config["external_encoder_dim"]
        one_hot_dim = 4
    elif config["task"] == "patterns_conditional_growth":
        use_one_hot_encoder = config["use_one_hot_encoder"]
        external_encoder_dim = config["external_encoder_dim"] if not use_one_hot_encoder else num_patterns
        seed_type = config["seed_type"]
        one_hot_dim = external_encoder_dim
        config["external_encoder_dim"] = external_encoder_dim
    else:
        raise ValueError(f"Unknown task: {config['task']}")

    # Create dataset based on task type
    if config["task"] == "patterns_morphing":
        # Comprehensive pattern morphing task with all combinations
        dataset = GoalPatternsMorphingDataset(
            size=pattern_size,
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            one_hot_encoder=use_one_hot_encoder,
            external_encoder_dim=one_hot_dim,  # Use one-hot dimension for dataset creation
            device=device,
            dtype=dtype,
            target_patterns=target_patterns,
            domain_noise=config["domain_noise"],
        )
        # If batch_size is None, use all combinations as a single batch
        if config["batch_size"] is None:
            batch_size = len(dataset)  # Use all combinations in one batch
        else:
            batch_size = config["batch_size"]  # Use configurable batch size

        dataset_checksum = GoalPatternsMorphingDataset(
            size=pattern_size,
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            one_hot_encoder=use_one_hot_encoder,
            external_encoder_dim=one_hot_dim,  # Use one-hot dimension for dataset creation
            device=device,
            dtype=dtype,
            target_patterns=target_patterns,
            domain_noise=0.0,
        )

    elif config["task"] == "patterns_rotation":
        raise NotImplementedError("Patterns rotation task is not implemented yet")
        # # New pattern transformation task
        # dataset = GoalPatternsTransformDataset(
        #     pattern_size=pattern_size,
        #     space_size=space_size,
        #     embedding_dim=embedding_dim,
        #     extra_channels=extra_channels,
        #     device=device,
        #     dtype=dtype,
        #     pattern_list=pattern_list,  # Pass pattern list from config
        #     transformation_amount=config["transformation_amount"],
        #     transformation_type="rotation",
        #     boundary_condition=config["boundary_condition"],
        #     num_samples_per_transformation=config["num_samples_per_transformation"],
        #     domain_noise=config["domain_noise"],
        # )
        # # Handle batch_size = None case
        # if config["batch_size"] is None:
        #     batch_size = len(dataset)  # Use all samples in one batch
        # else:
        #     batch_size = config["batch_size"]  # Use configurable batch size for larger dataset

    elif config["task"] == "patterns_translation":
        # New pattern translation task
        dataset = GoalPatternsTransformDataset(
            pattern_size=pattern_size,
            space_size=space_size,
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            device=device,
            dtype=dtype,
            target_patterns=target_patterns,
            transformation_amount=config["transformation_amount"],
            transformation_type="translation",
            boundary_condition=config["boundary_condition"],
            num_samples_per_transformation=config["num_samples_per_transformation"],
            domain_noise=config["domain_noise"],
            batch_size=config["batch_size"],
        )
        dataset_checksum = GoalPatternsTransformDataset(
            pattern_size=pattern_size,
            space_size=space_size,
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            device=device,
            dtype=dtype,
            target_patterns=target_patterns,
            transformation_amount=config["transformation_amount"],
            transformation_type="translation",
            boundary_condition=config["boundary_condition"],
            num_samples_per_transformation=1,
            domain_noise=0.0,
            batch_size=1,
        )
        # Handle batch_size = None case
        if config["batch_size"] is None:
            batch_size = len(dataset)  # Use all samples in one batch
        else:
            batch_size = config["batch_size"]  # Use configurable batch size for larger dataset

    elif config["task"] == "patterns_translation_trajectory":
        # New pattern translation trajectory task
        dataset = GoalPatternsTrajectoryDataset(
            pattern_size=pattern_size,
            space_size=space_size,
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            device=device,
            dtype=dtype,
            target_patterns=target_patterns,
            nca_steps=config["nca_steps"],  # Should be [min, max] for trajectory
            boundary_condition=config["boundary_condition"],
            num_samples_per_transformation=config["num_samples_per_transformation"],
            domain_noise=config["domain_noise"],
        )
        # Handle batch_size = None case
        if config["batch_size"] is None:
            batch_size = len(dataset)  # Use all samples in one batch
        else:
            batch_size = config["batch_size"]  # Use configurable batch size for larger dataset

    elif config["task"] == "patterns_conditional_growth":
        # Conditional growth task using GoalPatternsDataset with space_size canvas
        no_task_encoder = config["no_task_encoder"]
        seed_positions = config["seed_positions"]
        dataset = GoalPatternsDataset(
            size=pattern_size,
            seed_type=seed_type,
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            one_hot_encoder=use_one_hot_encoder,
            external_encoder_dim=one_hot_dim,  # Use one-hot dimension for dataset creation
            device=device,
            dtype=dtype,
            target_patterns=target_patterns,
            no_task_encoder=no_task_encoder,
            space_size=space_size,
            boundary_condition=config["boundary_condition"],
            seed_positions=seed_positions,
        )
        # If batch_size is None, use all patterns as a single batch
        if config["batch_size"] is None:
            batch_size = len(dataset)  # Use all patterns in one batch
        else:
            batch_size = config["batch_size"]  # Use configurable batch size

        # Create dataset_checksum for checksum evaluation (conditional growth doesn't need checksum eval, so use None)
        dataset_checksum = None

    else:
        raise ValueError(f"Unknown task: {config['task']}")

    dataset.train()

    # Use custom collate function for trajectory task to handle variable-length sequences
    # and for conditional growth with no task encoder to handle None values
    if config["task"] == "patterns_translation_trajectory":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.trajectory_collate_fn)
    elif config["task"] == "patterns_conditional_growth" and config["no_task_encoder"]:

        def conditional_growth_collate_fn(batch):
            """Custom collate function to handle None task encoders."""
            initial_seeds = []
            task_encoders = []
            targets = []

            for initial_seed, task_encoder, target in batch:
                initial_seeds.append(initial_seed)
                task_encoders.append(task_encoder)  # Will be None
                targets.append(target)

            # Stack tensors normally
            initial_seeds = torch.stack(initial_seeds, dim=0)
            targets = torch.stack(targets, dim=0)

            # For task encoders, since they're all None, just return None
            # (we don't need to stack None values)
            task_encoders = None

            return initial_seeds, task_encoders, targets

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=conditional_growth_collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if rank == 0:
        print(f"Dataset created with {len(dataset)} samples")

        # Save initial seeds as .pt file (like mNCA_singlePattern.py)
        # Sample a few seeds for saving
        sample_initial_seeds = []
        sample_targets = []
        sample_task_encoders = []

        # Get samples from dataset
        num_samples_to_save = min(16, len(dataset))  # Save up to 16 samples
        for i in range(num_samples_to_save):
            if config["task"] == "patterns_translation_trajectory":
                initial_seed, task_encoder, target_seq = dataset[i]
                # For trajectory, save the initial seed and first target
                sample_initial_seeds.append(initial_seed)
                sample_task_encoders.append(task_encoder)
                sample_targets.append(target_seq[0] if len(target_seq) > 0 else target_seq)
            else:
                initial_seed, task_encoder, target = dataset[i]
                sample_initial_seeds.append(initial_seed)
                sample_task_encoders.append(task_encoder)
                sample_targets.append(target)

        # Stack and save
        if sample_initial_seeds:
            initial_seeds_batch = torch.stack(sample_initial_seeds, dim=0)
            torch.save(initial_seeds_batch, config["_path"] + "/initial_seeds.pt")
            print(f"Saved initial seeds to: {config['_path']}/initial_seeds.pt")

            # Also save targets for reference
            targets_batch = torch.stack(sample_targets, dim=0)
            torch.save(targets_batch, config["_path"] + "/targets.pt")
            print(f"Saved targets to: {config['_path']}/targets.pt")

        if config["task"] == "patterns_morphing":
            print(f"Selected patterns: {dataset.pattern_identifiers}")
            print(
                f"Generated {len(dataset)} morphing combinations ({len(dataset.pattern_identifiers)}×{len(dataset.pattern_identifiers)})"
            )

            # Display the combinations matrix
            print("\nMorphing combinations matrix:")
            combinations_matrix = dataset.get_combinations_matrix()
            for row in combinations_matrix:
                print("  " + " | ".join(row))

            print(f"\nUsing one-hot encoder with dimension {external_encoder_dim}\n")

        elif config["task"] == "patterns_rotation":
            print(f"Selected patterns: {dataset.pattern_identifiers}")
            print(f"Task encoders: clockwise=[1,0], anticlockwise=[0,1], no_change=[0,0]")
            print(f"Using {len(dataset.pattern_identifiers)} patterns × 360 angles × 3 transformations (2D encoding)")
            print(f"Transformation type: rotation, amount: {dataset.transformation_amount}°")
            print(f"Boundary condition: {dataset.boundary_condition}\n")

        elif config["task"] == "patterns_translation":
            print(f"Selected patterns: {dataset.pattern_identifiers}")
            print(f"Task encoders: up=[1,0,0,0], right=[0,1,0,0], down=[0,0,1,0], left=[0,0,0,1], no_movement=[0,0,0,0]")
            print(f"Using {len(dataset.pattern_identifiers)} patterns × 5 transformations (4D encoding) × random positions")
            print(f"Transformation type: translation, amount: {dataset.transformation_amount}px")
            print(f"Boundary condition: {dataset.boundary_condition}\n")

        elif config["task"] == "patterns_translation_trajectory":
            print(f"Selected patterns: {dataset.pattern_identifiers}")
            print(f"Task encoders: up=[1,0,0,0], right=[0,1,0,0], down=[0,0,1,0], left=[0,0,0,1], no_movement=[0,0,0,0]")
            print(
                f"Using {len(dataset.pattern_identifiers)} patterns × {dataset.num_samples_per_transformation} trajectory samples"
            )
            if dataset.nca_steps_min == dataset.nca_steps_max:
                print(f"Trajectory length: {dataset.nca_steps_min} steps (fixed)")
            else:
                print(f"Trajectory lengths: [{dataset.nca_steps_min}, {dataset.nca_steps_max}] steps")
            print(f"Boundary condition: {dataset.boundary_condition}\n")

        elif config["task"] == "patterns_conditional_growth":
            print(f"Selected patterns: {dataset.pattern_identifiers}")
            print(f"Seed type: {seed_type}")
            print(f"Using one-hot encoder: {use_one_hot_encoder}")
            if use_one_hot_encoder:
                print(f"One-hot encoding with dimension {external_encoder_dim}")
            else:
                print(f"Random external encoders with dimension {external_encoder_dim}")
            print(f"Generated {len(dataset)} conditional growth samples (1 per pattern)")
            print(f"Task: Growing patterns from {seed_type} seeds using conditional vectors\n")

    # Parse activation functions
    activation_conv = return_activation_function(config["activation_conv"])
    activation_fc = return_activation_function(config["activation_fc"])
    activation_state_norm = config["state_norm"]
    activation_last = return_activation_function(config["activation_last"])
    activation_encoder = return_activation_function(config["activation_encoder"])

    # Create external encoder layer based on use_one_hot_encoder flag
    external_encoder_layer = None
    # Calculate task encoder channels
    task_encoder_channels = 0

    if config["task"] == "patterns_morphing":
        if config["pattern_size"] != config["space_size"]:
            raise ValueError("For patterns_morphing task, pattern_size must be equal to space_size")
        if config["use_one_hot_encoder"]:
            # Use one-hot directly
            task_encoder_channels = one_hot_dim
            if rank == 0:
                print(f"Using one-hot encoders directly with dimension: {one_hot_dim}")
        else:
            # Create external encoder layer
            if isinstance(external_encoder_dim, int):
                external_encoder_layer = torch.nn.Sequential(
                    torch.nn.Linear(one_hot_dim, external_encoder_dim, bias=config["bias"]),
                    activation_encoder,
                ).to(device)
                task_encoder_channels = external_encoder_dim
                if rank == 0:
                    print(f"Created external encoder layer: {one_hot_dim} -> {external_encoder_dim}")
            else:
                task_encoder_channels = one_hot_dim
                if rank == 0:
                    print(f"Using random encoders directly with dimension: {one_hot_dim}")
    elif config["task"] == "patterns_conditional_growth" and config["no_task_encoder"]:
        # For conditional growth with no task encoder, set task encoder channels to 0
        task_encoder_channels = 0
        if rank == 0:
            print("Using no task encoder for conditional growth - no task encoder channels")
    else:
        # For transform/translation tasks and conditional growth with task encoder, use_one_hot_encoder determines whether to use linear layer
        if config["use_one_hot_encoder"]:
            # Use one-hot directly
            task_encoder_channels = one_hot_dim
            if rank == 0:
                print(f"Using one-hot encoders directly with dimension: {one_hot_dim}")
        else:
            # Create linear layer if external_encoder_dim is specified
            if isinstance(external_encoder_dim, int):
                external_encoder_layer = torch.nn.Sequential(
                    torch.nn.Linear(one_hot_dim, external_encoder_dim, bias=config["bias"]),
                    activation_encoder,
                ).to(device)
                task_encoder_channels = external_encoder_dim
                if rank == 0:
                    print(f"Created external encoder layer: {one_hot_dim} -> {external_encoder_dim}")
            else:
                task_encoder_channels = one_hot_dim
                if rank == 0:
                    print(f"Using one-hot encoders directly with dimension: {one_hot_dim}")

    nca_external_channels = task_encoder_channels

    if rank == 0:
        print(f"Task encoder channels: {task_encoder_channels}")
        print(f"Total external channels: {nca_external_channels}")

    # Create model using config
    model = NCA_mlp(
        num_input_channels=embedding_dim + extra_channels,
        num_external_channels=nca_external_channels,
        num_output_conv_features=config["num_output_conv_features"],
        num_conv_layers=config["num_conv_layers"],
        hidden_dim_mlp=config["hidden_dim_mlp"],
        bias=config["bias"],
        activation_conv=activation_conv,
        activation_fc=activation_fc,
        activation_last=activation_last,
        stochastic_update_ratio=config["stochastic_update_ratio"],
        convolution_mode=config["convolution_mode"],
        fixed_kernels=config["fixed_kernels"],
        num_kernels=config["num_kernels"],
        custom_kernels=config["custom_kernels"],
        width_kernel=config["width_kernel"],
        additive_update=config["additive_update"],
        merge_ext=config["merge_ext"],
        dropout=config["dropout"],
        alive_mask_goal=alive_mask_goal,
        alive_threshold=alive_threshold,
        boundary_condition=config["boundary_condition"],
        custom_padding=config.get("custom_padding"),
        custom_padding_thickness=config.get("custom_padding_thickness"),
        embedding_dim=config.get("embedding_dim"),
        isotropic_only=config["isotropic_only"],
        extra_kernels=config["extra_kernels"],
    ).to(device)

    save_model_config_yaml(config["_path"] + "/model_config.yml", model, config)

    if rank == 0:
        print(model)
        nb_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float().numpy().shape[0]

        # Add external encoder parameters if it exists
        if external_encoder_layer is not None:
            encoder_params = (
                torch.nn.utils.parameters_to_vector(external_encoder_layer.parameters()).detach().cpu().float().numpy().shape[0]
            )
            nb_params += encoder_params
            print(f"External encoder parameters: {encoder_params}")

        print(f"\nTotal number of parameters: {nb_params}\n")
        config["nb_params"] = nb_params

        # Save external encoder configuration
        config["has_external_encoder"] = external_encoder_layer is not None
        config["one_hot_dim"] = one_hot_dim

        # Copy target pattern files to results folder
        targets_dir = os.path.join(config["_path"], "targets")
        os.makedirs(targets_dir, exist_ok=True)
        if isinstance(target_patterns, str) and target_patterns.endswith("/"):
            for f in os.listdir(target_patterns):
                if f.endswith(".png"):
                    src_path = os.path.join(target_patterns, f)
                    dst_path = os.path.join(targets_dir, f)
                    shutil.copy2(src_path, dst_path)
            print(f"Copied target patterns to: {targets_dir}")

        # Save comprehensive config for evaluation (like mNCA_singlePattern.py)
        eval_config = {
            # Core parameters
            "task": config["task"],
            "pattern_size": pattern_size,
            "space_size": space_size,
            "embedding_dim": embedding_dim,
            "extra_channels": extra_channels,
            "nca_steps": nca_steps_eval,
            "additive_update": config["additive_update"],
            "state_norm": config["state_norm"],
            # Model architecture
            "fixed_kernels": config["fixed_kernels"],
            "hidden_dim_mlp": config["hidden_dim_mlp"],
            "activation_conv": config["activation_conv"],
            "activation_fc": config["activation_fc"],
            "activation_last": config["activation_last"],
            "num_conv_layers": config["num_conv_layers"],
            "convolution_mode": config["convolution_mode"],
            "num_kernels": config["num_kernels"],
            "width_kernel": config["width_kernel"],
            "bias": config["bias"],
            # External encoder parameters
            "use_one_hot_encoder": config["use_one_hot_encoder"],
            "external_encoder_dim": config["external_encoder_dim"],
            "has_external_encoder": external_encoder_layer is not None,
            "one_hot_dim": one_hot_dim,
            "num_position_channels": 0,
            "num_goal_channels": task_encoder_channels,
            "total_external_channels": nca_external_channels,
            # Training parameters
            "seed_type": config["seed_type"],
            "batch_size": config["batch_size"],
            "device": str(device),
            "dtype": str(dtype),
            # Additional task-specific parameters
            "boundary_condition": config["boundary_condition"],
            "alive_mask_goal": config["alive_mask_goal"],
            "alive_mask": config["alive_mask"],
            "alive_threshold": alive_threshold,
            "seed_positions": config.get("seed_positions"),
            # Target patterns for evaluation
            "target_patterns": target_patterns,
        }

        # Save the comprehensive config
        with open(config["_path"] + "/eval_config.yml", "w") as f:
            yaml.dump(eval_config, f)
        print(f"Saved evaluation config to: {config['_path']}/eval_config.yml")

    # Wrap the model with DDP if distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    # Only watch the model on the master process
    if not distributed or torch.distributed.get_rank() == 0:
        wandb.watch(model, log=config["log_params"], log_freq=config["log_freq"])

    if external_encoder_layer is not None:
        # Add both model and encoder parameters to optimizer
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(external_encoder_layer.parameters()),
            lr=config["lr"],
            betas=tuple(config["betas"]),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]))

    criterion = torch.nn.MSELoss()

    # Setup learning rate scheduler
    scheduler = None
    if config["use_lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["lr_scheduler_factor"],
            patience=config["lr_scheduler_patience"],
            threshold=config["lr_scheduler_threshold"],
            min_lr=config["lr_scheduler_min_lr"],
            verbose=True if rank == 0 else False,
        )

    # Training loop - exactly matching original
    num_epochs = config["num_epochs"]
    best_loss = float("inf")

    # Initialize loss and learning rate tracking
    all_losses = []
    all_l1_checksums = []
    all_l1_checksums_eval = []
    all_learning_rates = []
    all_gradient_magnitudes = []

    # Create evaluation dataset once for checksum evaluation
    eval_dataset = None
    if config["task"] == "patterns_translation" and config["pattern_identifiers_eval"] is not None:
        eval_dataset = GoalPatternsTransformDataset(
            pattern_size=config["pattern_size"],
            space_size=config["space_size_eval"],
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            device=device,
            dtype=dtype,
            target_patterns=config["pattern_identifiers_eval"],
            transformation_amount=config["transformation_amount"],
            transformation_type="translation",
            boundary_condition=config["boundary_condition"],
            num_samples_per_transformation=1,
            domain_noise=0.0,  # No domain noise for evaluation
            batch_size=1,
        )
        eval_dataset.eval()

    tic = time.time()
    for epoch in tqdm(range(num_epochs), desc="Training NCA", disable=(rank != 0)):
        model.train()

        # Sample nca_steps for this epoch if it's a range
        if nca_steps_is_range:
            current_nca_steps = random.randint(nca_steps_min, nca_steps_max)
        else:
            current_nca_steps = nca_steps

        for batch_idx, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            # Handle different return formats for trajectory vs single-step tasks
            if config["task"] == "patterns_translation_trajectory":
                input_images, task_encoders, target_images, seq_lengths = batch_data
            else:
                input_images, task_encoders, target_images = batch_data

            state = input_images.clone().detach().requires_grad_(True)  # Reset the state for each batch

            # Get batch size for logging (needed in all branches)
            batch_size_actual = state.shape[0]

            if config["task"] == "patterns_translation_trajectory":
                # Handle trajectory task - task_encoders: [batch, seq_len, 4], target_images: [batch, seq_len, 3, H, W]
                _, num_channels, H, W = state.shape
                max_seq_len = task_encoders.shape[1]

                collected_states = []

                # Run NCA for seq_len steps with different task encoders at each step
                for step in range(max_seq_len):
                    # Get task encoder for current step
                    current_task_encoder = task_encoders[:, step, :]  # [batch, 4]

                    # Transform task encoders through external encoder layer if it exists
                    if external_encoder_layer is not None:
                        task_encoders_processed = external_encoder_layer(current_task_encoder)
                    else:
                        task_encoders_processed = current_task_encoder

                    task_encoders_spatial = prepare_external_inputs(
                        task_encoders_processed, batch_size_actual, H
                    )

                    # Run single NCA step
                    state = nca_step(
                        state,
                        model,
                        task_encoders_spatial,
                        config["update_noise"],
                        config["additive_update"],
                        config["alive_mask"],
                        config["state_norm"],
                    )

                    # Collect state for this step
                    collected_states.append(state[:, :embedding_dim])

                # Stack collected states to get [batch, seq_len, embedding_dim, H, W]
                predicted_sequence = torch.stack(collected_states, dim=1)

                # Create mask for valid positions (not padded)
                device = predicted_sequence.device
                batch_size = len(seq_lengths)
                mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
                for i, seq_len in enumerate(seq_lengths):
                    mask[i, :seq_len] = True

                # Compute loss only on valid (non-padded) positions
                valid_predicted = predicted_sequence[mask]  # [total_valid_steps, embedding_dim, H, W]
                valid_targets = target_images[mask]  # [total_valid_steps, embedding_dim, H, W]

                loss = criterion(valid_predicted, valid_targets)

            elif config["task"] == "patterns_translation":
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    task_encoders_processed = external_encoder_layer(task_encoders)
                else:
                    task_encoders_processed = task_encoders

                _, num_channels, H, W = state.shape
                task_encoders_spatial = prepare_external_inputs(task_encoders_processed, batch_size_actual, H)
                # Run the NCA for a randomly sampled number of steps
                for _ in range(current_nca_steps):
                    state = nca_step(
                        state,
                        model,
                        task_encoders_spatial,
                        config["update_noise"],
                        config["additive_update"],
                        config["alive_mask"],
                        config["state_norm"],
                    )

                loss = criterion(state[:, :embedding_dim], target_images[:, :embedding_dim])

            elif config["task"] == "patterns_morphing":
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    task_encoders_processed = external_encoder_layer(task_encoders)
                else:
                    task_encoders_processed = task_encoders

                _, num_channels, H, W = state.shape
                task_encoders_spatial = prepare_external_inputs(task_encoders_processed, batch_size_actual, H)

                # Run the NCA for a randomly sampled number of steps
                loss_all_dev = config.get("loss_all_dev", False)
                loss = 0.0
                loss_steps = 0
                for t in range(current_nca_steps):
                    state = nca_step(
                        state,
                        model,
                        task_encoders_spatial,
                        config["update_noise"],
                        config["additive_update"],
                        config["alive_mask"],
                        config["state_norm"],
                    )
                    if loss_all_dev and (not nca_steps_is_range or t >= nca_steps_min):
                        loss = loss + criterion(state[:, :embedding_dim], target_images[:, :embedding_dim])
                        loss_steps += 1
                loss = loss / loss_steps if loss_all_dev and loss_steps > 0 else criterion(state[:, :embedding_dim], target_images[:, :embedding_dim])

            elif config["task"] == "patterns_rotation":
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    task_encoders_processed = external_encoder_layer(task_encoders)
                else:
                    task_encoders_processed = task_encoders

                _, num_channels, H, W = state.shape
                task_encoders_spatial = prepare_external_inputs(task_encoders_processed, batch_size_actual, H)

                # Run the NCA for a randomly sampled number of steps
                for _ in range(current_nca_steps):
                    state = nca_step(
                        state,
                        model,
                        task_encoders_spatial,
                        config["update_noise"],
                        config["additive_update"],
                        config["alive_mask"],
                        config["state_norm"],
                    )

                loss = criterion(state[:, :embedding_dim], target_images[:, :embedding_dim])

            elif config["task"] == "patterns_conditional_growth":
                _, num_channels, H, W = state.shape

                if config["no_task_encoder"] or task_encoders is None:
                    task_encoders_spatial = None
                else:
                    if external_encoder_layer is not None:
                        task_encoders_processed = external_encoder_layer(task_encoders)
                    else:
                        task_encoders_processed = task_encoders

                    task_encoders_spatial = prepare_external_inputs(
                        task_encoders_processed, batch_size_actual, H
                    )

                # Run the NCA for a randomly sampled number of steps
                loss_all_dev = config.get("loss_all_dev", False)
                loss_all_dev_min_t = config.get("loss_all_dev_min_t", None)
                loss = 0.0
                loss_steps = 0
                for t in range(current_nca_steps):
                    state = nca_step(
                        state,
                        model,
                        task_encoders_spatial,
                        config["update_noise"],
                        config["additive_update"],
                        config["alive_mask"],
                        config["state_norm"],
                    )
                    if loss_all_dev and (not loss_all_dev_min_t or t >= loss_all_dev_min_t):
                        loss = loss + criterion(state[:, :embedding_dim], target_images[:, :embedding_dim])
                        loss_steps += 1
                loss = loss / loss_steps if loss_all_dev else criterion(state[:, :embedding_dim], target_images[:, :embedding_dim])

            else:
                raise ValueError(f"Unknown task: {config['task']}")

            loss.backward(retain_graph=False)

            # Compute gradient magnitude (10% of epochs)
            if rank == 0 and epoch % 10 == 0:
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                if external_encoder_layer is not None:
                    for p in external_encoder_layer.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm**0.5
                all_gradient_magnitudes.append(total_grad_norm)

            # Clip gradients (only if grad_clip is set to a value in config)
            grad_clip_value = config.get("grad_clip")
            if grad_clip_value is not None:
                if external_encoder_layer is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(external_encoder_layer.parameters()),
                        grad_clip_value,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()

            if epoch % 100 == 0 and batch_idx == 0 and rank == 0:
                # Periodic checksum evaluation for patterns_translation task
                checksum_l1 = evaluate_checksum_l1_distance(
                    config,
                    model,
                    # dataset,
                    # [dataset[0]],  # only one sample to keep memory usage low
                    dataset_checksum,  # remove noisy variations and have 0 initial state noise
                    external_encoder_layer,
                    activation_state_norm,
                    device,  # "cpu",
                )
                all_l1_checksums.append(checksum_l1)

                # Periodic checksum evaluation for evaluation patterns
                checksum_l1_eval = None
                if eval_dataset is not None:
                    checksum_l1_eval = evaluate_checksum_l1_distance(
                        config,
                        model,
                        eval_dataset,  # [eval_dataset[0]],  # only one sample to keep memory usage low
                        external_encoder_layer,
                        activation_state_norm,
                        device,  # "cpu",
                    )
                all_l1_checksums_eval.append(checksum_l1_eval)
                model.to(device)

                current_lr = optimizer.param_groups[0]["lr"]
                # if checksum_l1_eval is None, set it to 0
                if checksum_l1_eval is None:
                    tqdm.write(
                        f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}, NCA Steps: {current_nca_steps}, Batch size: {batch_size_actual}"
                    )
                else:
                    tqdm.write(
                        f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}, NCA Steps: {current_nca_steps}, Batch size: {batch_size_actual}, Checksum train: {checksum_l1:.1f}, Checksum eval: {checksum_l1_eval:.1f}"
                    )
                    wandb.log(
                        {
                            "checksum_l1": checksum_l1,
                            "checksum_l1_eval": checksum_l1_eval,
                            "epoch": epoch,
                        }
                    )

        if rank == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            # Log to wandb with NCA steps information
            if nca_steps_is_range:
                wandb.log({"train_loss": current_loss, "epoch": epoch, "nca_steps": current_nca_steps})
            else:
                wandb.log({"train_loss": current_loss, "epoch": epoch})
            all_losses.append(current_loss)
            all_learning_rates.append(current_lr)

            if current_loss < best_loss:
                best_loss = current_loss
                model_to_save = model.module if distributed else model
                torch.save(model_to_save.state_dict(), config["_path"] + "/best_model.pth")

                # Save external encoder layer if it exists
                if external_encoder_layer is not None:
                    torch.save(
                        external_encoder_layer.state_dict(),
                        config["_path"] + "/best_external_encoder.pth",
                    )

            # Step the learning rate scheduler
            if scheduler is not None:
                scheduler.step(current_loss)
                # Log current learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"learning_rate": current_lr, "epoch": epoch})

    if rank == 0:
        print("Training took:", time.time() - tic)

        # Save training curve and losses
        all_losses = np.array(all_losses)
        all_learning_rates = np.array(all_learning_rates)
        all_l1_checksums = np.array(all_l1_checksums) if all_l1_checksums else np.array([])
        all_l1_checksums_eval = np.array(all_l1_checksums_eval) if all_l1_checksums_eval else np.array([])
        all_gradient_magnitudes = np.array(all_gradient_magnitudes)

        # Save numpy arrays with losses, learning rates, and checksum L1 distances
        np.save(f"{config['_path']}/training_losses_{config['id']}.npy", all_losses)
        np.save(f"{config['_path']}/training_learning_rates_{config['id']}.npy", all_learning_rates)
        np.save(f"{config['_path']}/gradient_magnitudes_{config['id']}.npy", all_gradient_magnitudes)
        if len(all_l1_checksums) > 0:
            np.save(f"{config['_path']}/checksum_l1_distances_{config['id']}.npy", all_l1_checksums)
        if len(all_l1_checksums_eval) > 0:
            np.save(
                f"{config['_path']}/checksum_l1_distances_eval_{config['id']}.npy",
                all_l1_checksums_eval,
            )

        # Create and save training curve
        plot_training_curve(all_losses, all_learning_rates, config["id"], config["_path"])

        # Create and save gradient magnitudes curve
        plot_gradient_magnitudes(all_gradient_magnitudes, config["id"], config["_path"])
        print(f"Gradient magnitudes curve saved to {config['_path']}/gradient_magnitudes_{config['id']}.pdf")

        # Create and save checksum L1 curve
        if config["task"] == "patterns_translation":
            plot_checksum_l1_curve(all_l1_checksums, all_l1_checksums_eval, config["id"], config["_path"])
            print(f"Checksum L1 curve saved to {config['_path']}/checksum_l1_curve_{config['id']}.pdf")

        print(f"Training curve and losses saved to {config['_path']}")

    # Evaluation and visualization
    if rank == 0:
        # Load best model
        model_to_load = model.module if distributed else model
        model_to_load.load_state_dict(torch.load(config["_path"] + "/best_model.pth", map_location=device))

        # Load best external encoder layer if it exists
        if external_encoder_layer is not None:
            external_encoder_layer.load_state_dict(
                torch.load(config["_path"] + "/best_external_encoder.pth", map_location=device)
            )

        # Create animations directories
        skip_animations = "skip_animations" in config and config["skip_animations"]
        animations_path = os.path.join(config["_path"], "animations", "train")
        animations_eval_path = os.path.join(config["_path"], "animations", "eval")
        if skip_animations:
            _noop = lambda *a, **kw: []
            animate_states = _noop
            animate_hidden_channels = _noop
            create_animation_grid = _noop
            create_transform_animation = _noop
            wandb.Video = lambda *a, **kw: None
        else:
            os.makedirs(animations_path, exist_ok=True)
            os.makedirs(animations_eval_path, exist_ok=True)
            from src.visualisation.viz import animate_states, animate_hidden_channels
            from NCAs.visualisation_functions import (
                create_animation_grid,
                create_transform_animation,
            )

        dataset.eval()

        if config["task"] == "patterns_morphing":
            if config["space_size"] != config["pattern_size"]:
                raise ValueError("patterns_morphing task requires space_size to be equal to pattern_size")

            # For animation, organize samples by source pattern (each row = one source pattern)
            num_patterns = len(dataset.pattern_identifiers)

            # Use all combinations - grid size matches number of patterns
            sample_indices = list(range(len(dataset)))
            num_samples = len(dataset)

            # Batch selected samples
            all_initial_seeds = []
            all_task_encoders = []
            for idx in sample_indices:
                initial_seed, task_encoder, target = dataset[idx]
                all_initial_seeds.append(initial_seed)
                all_task_encoders.append(task_encoder)
            all_initial_seeds = torch.stack(all_initial_seeds, dim=0).to(device)  # [batch, C, H, W]
            all_task_encoders = torch.stack(all_task_encoders, dim=0).to(device)  # [batch, encoder_dim]

            # Run evaluate_nca in batch
            with torch.no_grad():
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    all_task_encoders_processed = external_encoder_layer(all_task_encoders)
                else:
                    all_task_encoders_processed = all_task_encoders

                final_step, results = evaluate_nca_batched(
                    model_to_load.to(device),
                    all_initial_seeds.to(device),
                    (all_task_encoders_processed.to(device) if all_task_encoders_processed is not None else None),
                    nca_steps_eval,
                    additive=config["additive_update"],
                    state_norm=activation_state_norm,
                    alive_mask=config["alive_mask"],
                    update_noise=0.0,
                )

            # Save the grid image to the results folder
            # For organized layout: rows = source patterns, cols = target patterns
            grid_rows = num_patterns
            grid_cols = num_patterns
            total_cells = grid_rows * grid_cols

            # Prepare arrays for sources, targets and results
            source_images = []
            target_images = []
            result_images = []

            for i, sample_idx in enumerate(sample_indices):
                initial_seed, _, target = dataset[sample_idx]
                sample_info = dataset.get_sample_info(sample_idx)

                # Get source image from initial seed
                source = initial_seed[:embedding_dim].cpu().numpy().transpose(1, 2, 0)
                target = target[:embedding_dim].cpu().numpy().transpose(1, 2, 0)
                result = final_step[i, :embedding_dim].cpu().numpy().clip(0, 1).transpose(1, 2, 0)

                source_images.append(source)
                target_images.append(target)
                result_images.append(result)

            # Create morphing grid
            plot_morphing_grid(
                source_images,
                target_images,
                result_images,
                dataset,
                sample_indices,
                config["id"],
                config["_path"],
            )

            # Create grid animation of NCA evolution for morphing samples - save to main results folder
            id_ = config["id"]

            # Create dataset for animation with properly organized samples
            class SampledDataset:
                def __init__(self, selected_patterns, samples):
                    self.pattern_identifiers = selected_patterns
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return (
                        self.samples[idx]["initial_seed"],
                        self.samples[idx]["task_encoder"],
                        self.samples[idx]["target"],
                    )

            sampled_dataset_for_animation = SampledDataset(
                dataset.pattern_identifiers, [dataset.samples[i] for i in sample_indices]
            )

            create_animation_grid(
                sampled_dataset_for_animation,
                embedding_dim,
                results.cpu().numpy(),
                output_path=animations_path,
                id_=id_,
            )

            # Save morphing animation to wandb as video
            wandb.log({"morphing_animation_train": wandb.Video(f"{animations_path}/all_patterns_grid_anim_{id_}.mp4")})

            # Create evaluation dataset animations with pattern_list_eval
            if config["pattern_identifiers_eval"] is not None and config["pattern_identifiers_eval"]:
                print("Creating morphing animation for evaluation patterns...")
                # Create morphing dataset with evaluation patterns

                dataset_eval_patterns = GoalPatternsMorphingDataset(
                    size=pattern_size,
                    embedding_dim=embedding_dim,
                    extra_channels=extra_channels,
                    one_hot_encoder=config["use_one_hot_encoder"],
                    external_encoder_dim=one_hot_dim,
                    device=device,
                    dtype=dtype,
                    target_patterns=config["pattern_identifiers_eval"],
                    domain_noise=0.0,  # No domain noise for evaluation
                )
                dataset_eval_patterns.eval()

                # Run evaluation on eval patterns
                num_eval_patterns = len(config["pattern_identifiers_eval"])
                eval_sample_indices = list(range(len(dataset_eval_patterns)))

                # Batch eval samples
                eval_initial_seeds = []
                eval_task_encoders = []
                for idx in eval_sample_indices:
                    initial_seed, task_encoder, target = dataset_eval_patterns[idx]
                    eval_initial_seeds.append(initial_seed)
                    eval_task_encoders.append(task_encoder)
                eval_initial_seeds = torch.stack(eval_initial_seeds, dim=0).to(device)
                eval_task_encoders = torch.stack(eval_task_encoders, dim=0).to(device)

                # Run evaluate_nca in batch for eval patterns
                with torch.no_grad():
                    if external_encoder_layer is not None:
                        eval_task_encoders_processed = external_encoder_layer(eval_task_encoders)
                    else:
                        eval_task_encoders_processed = eval_task_encoders

                    eval_final_step, eval_results = evaluate_nca_batched(
                        model_to_load.to("cpu"),
                        eval_initial_seeds.cpu(),
                        (eval_task_encoders_processed.cpu() if eval_task_encoders_processed is not None else None),
                        nca_steps_eval,
                        additive=config["additive_update"],
                        state_norm=activation_state_norm,
                        alive_mask=config["alive_mask"],
                        update_noise=0.0,
                    )

                # Create dataset wrapper for animation
                class EvalSampledDataset:
                    def __init__(self, selected_patterns, samples):
                        self.pattern_identifiers = selected_patterns
                        self.samples = samples

                    def __len__(self):
                        return len(self.samples)

                    def __getitem__(self, idx):
                        return (
                            self.samples[idx]["initial_seed"],
                            self.samples[idx]["task_encoder"],
                            self.samples[idx]["target"],
                        )

                eval_sampled_dataset_for_animation = EvalSampledDataset(
                    config["pattern_identifiers_eval"],
                    [dataset_eval_patterns.samples[i] for i in eval_sample_indices],
                )

                create_animation_grid(
                    eval_sampled_dataset_for_animation,
                    embedding_dim,
                    eval_results.cpu().numpy(),
                    output_path=animations_eval_path,
                    id_=f"{config['id']}_eval_patterns",
                )

                # Save evaluation morphing animation to wandb as video
                wandb.log(
                    {
                        "morphing_animation_eval_patterns": wandb.Video(
                            f"{animations_eval_path}/all_patterns_grid_anim_{config['id']}_eval_patterns.mp4"
                        )
                    }
                )

        elif config["task"] == "patterns_rotation":
            # Transformation task visualization
            # Sample a few examples from the dataset to visualize
            num_samples = min(16, len(dataset))
            sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            # Collect samples for visualization
            input_images = []
            task_encoders = []
            target_images = []

            for idx in sample_indices:
                input_img, task_enc, target_img = dataset[idx]
                input_images.append(input_img)
                task_encoders.append(task_enc)
                target_images.append(target_img)

            input_images = torch.stack(input_images, dim=0).to(device)
            task_encoders = torch.stack(task_encoders, dim=0).to(device)
            target_images = torch.stack(target_images, dim=0).to(device)

            # Run NCA on samples
            with torch.no_grad():
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    task_encoders_processed = external_encoder_layer(task_encoders)
                else:
                    task_encoders_processed = task_encoders

                final_step, results = evaluate_nca_batched(
                    model_to_load.to("cpu"),
                    input_images.cpu(),
                    task_encoders_processed.cpu(),
                    nca_steps_eval,
                    additive=config["additive_update"],
                    state_norm=activation_state_norm,
                    alive_mask=config["alive_mask"],
                    update_noise=0.0,
                )

            # Prepare data for plotting
            input_imgs = []
            target_imgs = []
            result_imgs = []

            for i in range(num_samples):
                input_img = input_images[i, :embedding_dim].cpu().numpy().transpose(1, 2, 0)
                target_img = target_images[i, :embedding_dim].cpu().numpy().transpose(1, 2, 0)
                result_img = final_step[i, :embedding_dim].clip(0, 1).detach().cpu().numpy().transpose(1, 2, 0)

                input_imgs.append(input_img)
                target_imgs.append(target_img)
                result_imgs.append(result_img)

            # Create transformations grid
            plot_transformations_grid(
                input_imgs,
                target_imgs,
                result_imgs,
                dataset,
                sample_indices,
                config["id"],
                config["_path"],
                "rotation",
            )

            # Create rotation animation showing full 360-degree rotation
            print("Creating transform animation with training space_size...")
            animation_files_train = create_transform_animation(
                config=config,
                model=model_to_load,
                dataset=dataset,
                embedding_dim=embedding_dim,
                nca_steps=nca_steps_eval,
                output_path=animations_path,
                device=device,
                id_=f"{config['id']}_train_space",
                external_encoder_layer=external_encoder_layer,
            )

            # Save training rotation animation to wandb as video
            if animation_files_train:
                wandb.log({"rotation_animation_train": wandb.Video(animation_files_train[0])})

            # Create evaluation dataset animations with pattern_list_eval
            if config["pattern_identifiers_eval"] is not None and config["pattern_identifiers_eval"]:
                print("Creating rotation animation for evaluation patterns...")
                # Create rotation dataset with evaluation patterns
                eval_dataset.eval()

                create_transform_animation(
                    config=config,
                    model=model_to_load,
                    dataset=eval_dataset,
                    embedding_dim=embedding_dim,
                    nca_steps=nca_steps_eval,
                    output_path=animations_eval_path,
                    device=device,
                    id_=f"{config['id']}_eval_patterns",
                    external_encoder_layer=external_encoder_layer,
                )

                # Save evaluation pattern rotation animation to wandb as video
                wandb.log(
                    {
                        "rotation_animation_eval_patterns": wandb.Video(
                            f"{animations_eval_path}/rotation_animation_{config['id']}_eval_patterns.mp4"
                        )
                    }
                )

        elif config["task"] == "patterns_translation":
            # Translation task visualization
            # Sample a few examples from the dataset to visualize
            num_samples = min(16, len(dataset))
            sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            # Collect samples for visualization
            input_images = []
            task_encoders = []
            target_images = []

            for idx in sample_indices:
                input_img, task_enc, target_img = dataset[idx]
                input_images.append(input_img)
                task_encoders.append(task_enc)
                target_images.append(target_img)

            input_images = torch.stack(input_images, dim=0).to(device)
            task_encoders = torch.stack(task_encoders, dim=0).to(device)
            target_images = torch.stack(target_images, dim=0).to(device)

            # Run NCA on samples
            with torch.no_grad():
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    task_encoders_processed = external_encoder_layer(task_encoders)
                else:
                    task_encoders_processed = task_encoders

                final_step, results = evaluate_nca_batched(
                    model_to_load.to("cpu"),
                    input_images.cpu(),
                    task_encoders_processed.cpu(),
                    nca_steps_eval,
                    additive=config["additive_update"],
                    state_norm=activation_state_norm,
                    alive_mask=config["alive_mask"],
                    update_noise=0.0,
                )

            # Prepare data for plotting
            input_imgs = []
            target_imgs = []
            result_imgs = []

            for i in range(num_samples):
                input_img = input_images[i, :embedding_dim].cpu().numpy().transpose(1, 2, 0)
                target_img = target_images[i, :embedding_dim].cpu().numpy().transpose(1, 2, 0)
                result_img = final_step[i, :embedding_dim].clip(0, 1).detach().cpu().numpy().transpose(1, 2, 0)

                input_imgs.append(input_img)
                target_imgs.append(target_img)
                result_imgs.append(result_img)

            # Create transformations grid
            plot_transformations_grid(
                input_imgs,
                target_imgs,
                result_imgs,
                dataset,
                sample_indices,
                config["id"],
                config["_path"],
                "translation",
            )

            # Create translation animation
            print("Creating transform animation with training space_size...")
            animation_files = create_transform_animation(
                config=config,
                model=model_to_load,
                dataset=dataset,
                embedding_dim=embedding_dim,
                nca_steps=nca_steps_eval,
                output_path=animations_path,
                device=device,
                id_=f"{config['id']}_train_space",
                external_encoder_layer=external_encoder_layer,
            )

            # Save training translation animation files to wandb as videos
            for animation_file in animation_files:
                # Extract direction from filename
                direction = animation_file.split("/")[-1].split("_")[1]  # e.g., "up", "right", etc.
                wandb.log({f"translation_animation_{direction}_train": wandb.Video(animation_file)})

            # Create evaluation dataset with different space_size if specified
            if config["space_size_eval"] != config["space_size"] and eval_dataset is not None:
                print(f"Creating transform animation with evaluation space_size ({config['space_size_eval']})...")
                eval_dataset.eval()
                animation_files_eval = create_transform_animation(
                    config=config,
                    model=model_to_load,
                    dataset=eval_dataset,
                    embedding_dim=embedding_dim,
                    nca_steps=nca_steps_eval,
                    output_path=animations_eval_path,
                    device=device,
                    id_=f"{config['id']}_eval_space",
                    external_encoder_layer=external_encoder_layer,
                )

                # Save evaluation translation animation files to wandb as videos
                for animation_file in animation_files_eval:
                    # Extract direction from filename
                    direction = animation_file.split("/")[-1].split("_")[1]  # e.g., "up", "right", etc.
                    wandb.log({f"translation_animation_{direction}_eval_space": wandb.Video(animation_file)})

            # Create evaluation dataset animations with pattern_list_eval
            if config["pattern_identifiers_eval"] is not None and eval_dataset is not None:
                print("Creating transform animation for evaluation patterns...")
                animation_files_eval_patterns = create_transform_animation(
                    config=config,
                    model=model_to_load,
                    dataset=eval_dataset,
                    embedding_dim=embedding_dim,
                    nca_steps=nca_steps_eval,
                    output_path=animations_eval_path,
                    device=device,
                    id_=f"{config['id']}_eval_patterns",
                    external_encoder_layer=external_encoder_layer,
                )

                # Save evaluation pattern animation files to wandb as videos
                for animation_file in animation_files_eval_patterns:
                    # Extract direction from filename
                    direction = animation_file.split("/")[-1].split("_")[1]  # e.g., "up", "right", etc.
                    wandb.log({f"translation_animation_{direction}_eval_patterns": wandb.Video(animation_file)})

        elif config["task"] == "patterns_translation_trajectory":
            # Trajectory task evaluation
            # Sample a few trajectory examples from the dataset
            num_samples = min(8, len(dataset))
            sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            # Collect samples for evaluation
            input_images = []
            task_encoders_sequences = []
            target_images_sequences = []
            seq_lengths = []

            for idx in sample_indices:
                input_img, task_enc_seq, target_img_seq = dataset[idx]
                input_images.append(input_img)
                task_encoders_sequences.append(task_enc_seq)
                target_images_sequences.append(target_img_seq)
                seq_lengths.append(task_enc_seq.shape[0])

            # Use the trajectory collate function to handle variable lengths
            eval_batch = list(zip(input_images, task_encoders_sequences, target_images_sequences))
            input_images, task_encoders_sequences, target_images_sequences, seq_lengths = dataset.trajectory_collate_fn(
                eval_batch
            )

            input_images = input_images.to(device)
            task_encoders_sequences = task_encoders_sequences.to(device)
            target_images_sequences = target_images_sequences.to(device)

            # Run NCA on trajectory samples
            with torch.no_grad():
                batch_size_actual, num_channels, H, W = input_images.shape
                max_seq_len = task_encoders_sequences.shape[1]

                state = input_images.clone()
                collected_states = []

                # Run NCA for seq_len steps with different task encoders at each step
                for step in range(max_seq_len):
                    # Get task encoder for current step
                    current_task_encoder = task_encoders_sequences[:, step, :]  # [batch, 4]

                    # Transform task encoders through external encoder layer if it exists
                    if external_encoder_layer is not None:
                        task_encoders_processed = external_encoder_layer(current_task_encoder)
                    else:
                        task_encoders_processed = current_task_encoder

                    task_encoders_spatial = prepare_external_inputs(
                        task_encoders_processed, batch_size_actual, H
                    )

                    # Run single NCA step
                    state = nca_step(
                        state,
                        model_to_load,
                        task_encoders_spatial,
                        0.0,
                        config["additive_update"],
                        config["alive_mask"],
                        config["state_norm"],
                    )

                    # Collect state for this step
                    collected_states.append(state[:, :embedding_dim])

                # Stack collected states to get [batch, seq_len, embedding_dim, H, W]
                predicted_sequence = torch.stack(collected_states, dim=1)

                # Create mask for valid positions (not padded)
                device = predicted_sequence.device
                batch_size = len(seq_lengths)
                mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
                for i, seq_len in enumerate(seq_lengths):
                    mask[i, :seq_len] = True

                # Compute evaluation loss only on valid positions
                valid_predicted = predicted_sequence[mask]
                valid_targets = target_images_sequences[mask]
                eval_loss = criterion(valid_predicted, valid_targets)

                # Log evaluation metrics
                avg_seq_len = sum(seq_lengths) / len(seq_lengths)
                wandb.log(
                    {
                        "eval_trajectory_loss": eval_loss.item(),
                        "eval_avg_trajectory_length": avg_seq_len,
                        "eval_samples": num_samples,
                    }
                )

                print(f"Trajectory evaluation loss: {eval_loss.item():.6f}")
                print(f"Average trajectory length: {avg_seq_len:.1f}")

                # Create trajectory animation
                print("Creating trajectory animation...")

                # Convert collected_states from [batch, seq_len, embedding_dim, H, W] to [seq_len, batch, embedding_dim, H, W]
                # We need to stack the original collected states before they were used for loss calculation
                trajectory_states = torch.stack(collected_states, dim=0)  # [seq_len, batch, embedding_dim, H, W]

                # Convert to numpy for animation
                trajectory_states_np = trajectory_states.cpu().numpy()

                # Create a simple dataset wrapper for animation
                class TrajectoryDataset:
                    def __init__(self, dataset, sample_indices):
                        self.dataset = dataset
                        self.sample_indices = sample_indices
                        # Map sample indices to pattern names using the modulo operation
                        self.pattern_identifiers = [
                            dataset.pattern_identifiers[idx % dataset.num_patterns] for idx in sample_indices
                        ]

                    def __len__(self):
                        return len(self.sample_indices)

                    def __getitem__(self, idx):
                        # Return the original dataset item for the sampled index
                        return self.dataset[self.sample_indices[idx]]

                trajectory_dataset_wrapper = TrajectoryDataset(dataset, sample_indices)

                # Create animation grid
                create_animation_grid(
                    trajectory_dataset_wrapper,
                    embedding_dim,
                    trajectory_states_np,
                    output_path=animations_path,
                    id_=f"{config['id']}_trajectory",
                )

                # Save trajectory animation to wandb as video
                wandb.log(
                    {
                        "trajectory_animation_train": wandb.Video(
                            f"{animations_path}/all_patterns_grid_anim_{config['id']}_trajectory.mp4"
                        )
                    }
                )

                print(f"Trajectory animation saved to {animations_path}/all_patterns_grid_anim_{config['id']}_trajectory.mp4")

                # Create single-action trajectory evaluation (batch of 5, one per action)
                print("Creating single-action trajectory evaluation...")

                # Define all possible actions
                single_actions = [
                    ("up", torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dataset.dtype, device=device)),
                    (
                        "down",
                        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=dataset.dtype, device=device),
                    ),
                    (
                        "left",
                        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dataset.dtype, device=device),
                    ),
                    (
                        "right",
                        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=dataset.dtype, device=device),
                    ),
                    (
                        "stay",
                        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=dataset.dtype, device=device),
                    ),
                ]

                # Create batch of 5 samples, one per action (using first pattern)
                pattern_tensor = dataset.pattern_tensors[0]  # Use first pattern
                trajectory_length = 30

                # Create input images (5 identical starting positions)
                center_x = dataset.space_size // 2
                center_y = dataset.space_size // 2
                input_image = dataset._embed_pattern_in_space(pattern_tensor, center_x, center_y)

                # Add extra channels if needed
                if dataset.extra_channels > 0:
                    extra_padding = torch.zeros(
                        dataset.extra_channels,
                        dataset.space_size,
                        dataset.space_size,
                        dtype=dataset.dtype,
                        device=dataset.device,
                    )
                    input_image = torch.cat([input_image, extra_padding], dim=0)

                # Create batch of 5 identical input images
                input_images_batch = input_image.unsqueeze(0).repeat(5, 1, 1, 1).to(device)

                # Create task encoders batch: each sample uses the same action for all steps
                task_encoders_batch = torch.zeros(5, trajectory_length, 4, dtype=dataset.dtype, device=device)
                for i, (action_name, action_encoder) in enumerate(single_actions):
                    # Fill all steps with the same action
                    task_encoders_batch[i, :, :] = action_encoder

                # Run NCA on single-action batch
                with torch.no_grad():
                    batch_size_actual, num_channels, H, W = input_images_batch.shape
                    max_seq_len = task_encoders_batch.shape[1]

                    state = input_images_batch.clone()
                    collected_states = []

                    # Run NCA for seq_len steps
                    for step in range(max_seq_len):
                        # Get task encoder for current step
                        current_task_encoder = task_encoders_batch[:, step, :]  # [5, 4]

                        # Transform task encoders through external encoder layer if it exists
                        if external_encoder_layer is not None:
                            task_encoders_processed = external_encoder_layer(current_task_encoder)
                        else:
                            task_encoders_processed = current_task_encoder

                        task_encoders_spatial = prepare_external_inputs(
                            task_encoders_processed, batch_size_actual, H
                        )

                        # Run single NCA step
                        state = nca_step(
                            state,
                            model_to_load,
                            task_encoders_spatial,
                            0.0,
                            config["additive_update"],
                            config["alive_mask"],
                            config["state_norm"],
                        )

                        # Collect state for this step
                        collected_states.append(state[:, :embedding_dim])

                    # Convert collected_states to the format expected by create_animation_grid
                    single_actions_states = torch.stack(collected_states, dim=0)  # [seq_len, 5, embedding_dim, H, W]
                    single_actions_states_np = single_actions_states.cpu().numpy()

                    # Create simple dataset wrapper for the 5 actions
                    class SingleActionsDataset:
                        def __init__(self, actions):
                            self.actions = actions
                            self.pattern_identifiers = [f"🦎_{action[0]}" for action in actions]

                        def __len__(self):
                            return len(self.actions)

                        def __getitem__(self, idx):
                            # Return dummy data since we only need the interface
                            return None, None, None

                    single_actions_dataset = SingleActionsDataset(single_actions)

                    # Create animation grid for all 5 actions
                    create_animation_grid(
                        single_actions_dataset,
                        embedding_dim,
                        single_actions_states_np,
                        output_path=animations_path,
                        id_=f"{config['id']}_single_actions",
                    )

                    # Save single-actions animation to wandb
                    animation_file = f"{animations_path}/all_patterns_grid_anim_{config['id']}_single_actions.mp4"
                    wandb.log({"single_actions_animation_train": wandb.Video(animation_file)})

                    print(f"Single-actions trajectory animation saved to {animation_file}")
                    print("Grid shows: up, down, left, right, stay (in that order)")

        elif config["task"] == "patterns_conditional_growth":
            # Conditional growth task visualization
            # Use all samples from the dataset
            num_samples = len(dataset)
            sample_indices = list(range(num_samples))

            # Batch all samples
            all_initial_seeds = []
            all_task_encoders = []
            all_targets = []

            for idx in sample_indices:
                initial_seed, task_encoder, target = dataset[idx]
                all_initial_seeds.append(initial_seed)
                all_task_encoders.append(task_encoder)
                all_targets.append(target)

            all_initial_seeds = torch.stack(all_initial_seeds, dim=0).to(device)
            all_targets = torch.stack(all_targets, dim=0).to(device)

            # Handle task encoders - they might be None if no_task_encoder is True
            if config["no_task_encoder"]:
                all_task_encoders_processed = None
            else:
                all_task_encoders = torch.stack(all_task_encoders, dim=0).to(device)
                # Transform task encoders through external encoder layer if it exists
                if external_encoder_layer is not None:
                    all_task_encoders_processed = external_encoder_layer(all_task_encoders)
                else:
                    all_task_encoders_processed = all_task_encoders

            # Run evaluate_nca in batch
            with torch.no_grad():

                final_step, results = evaluate_nca_batched(
                    model_to_load.to(device),
                    all_initial_seeds.to(device),
                    (all_task_encoders_processed.to(device) if all_task_encoders_processed is not None else None),
                    nca_steps_eval,
                    additive=config["additive_update"],
                    state_norm=activation_state_norm,
                    alive_mask=config["alive_mask"],
                    update_noise=0.0,
                )

            # Create MP4 animations for main training results (like mNCA_evaluate_trained_model.py)
            print("Creating MP4 animations...")
            animations_dir = config["_path"] + "/animations/train"
            pathlib.Path(animations_dir).mkdir(parents=True, exist_ok=True)

            # results shape: [time_steps, batch_size, channels, H, W]
            num_train_samples = results.shape[1]

            for pattern_idx in range(num_train_samples):
                pattern_name = (
                    dataset.pattern_identifiers[pattern_idx]
                    if pattern_idx < len(dataset.pattern_identifiers)
                    else f"sample_{pattern_idx}"
                )

                # Main pattern animation (embedding channels only)
                states_to_viz = results[:, pattern_idx, :embedding_dim, :, :].detach().cpu().numpy().clip(0, 1)

                visualise_every_n_steps = 1
                animation_path = f"{animations_dir}/nca_grown_{pattern_name}_pattern_{pattern_idx}_conditional_growth.mp4"
                animate_states(
                    states_to_viz,
                    frames=range(0, states_to_viz.shape[0], visualise_every_n_steps),
                    interval=200,
                    cmap="gray" if embedding_dim == 1 else None,
                    filename=animation_path,
                )
                print(f"Saved animation: {animation_path}")

                # Hidden channels animation (if there are extra channels)
                nb_total_channels = embedding_dim + extra_channels
                if nb_total_channels > embedding_dim:
                    hidden_states_to_viz = results[:, pattern_idx, embedding_dim:, :, :].detach().cpu().numpy()

                    hidden_animation_path = (
                        f"{animations_dir}/hidden_channels_{pattern_name}_pattern_{pattern_idx}_conditional_growth.mp4"
                    )
                    animate_hidden_channels(
                        hidden_states_to_viz,
                        frames=range(0, hidden_states_to_viz.shape[0], visualise_every_n_steps),
                        interval=200,
                        filename=hidden_animation_path,
                    )
                    print(f"Saved hidden channels animation: {hidden_animation_path}")

            # Prepare data for plotting
            seed_images = []
            target_images = []
            result_images = []

            for i in range(num_samples):
                # Get seed image (initial state)
                seed_img = all_initial_seeds[i, :embedding_dim].cpu().numpy().transpose(1, 2, 0)
                # Get target image
                target_img = all_targets[i, :embedding_dim].cpu().numpy().transpose(1, 2, 0)
                # Get result image
                result_img = final_step[i, :embedding_dim].cpu().numpy().clip(0, 1).transpose(1, 2, 0)

                seed_images.append(seed_img)
                target_images.append(target_img)
                result_images.append(result_img)

            # Save hidden states visualizations (like mNCA_evaluate_trained_model.py)
            if extra_channels > 0:
                print("Creating hidden states visualizations...")
                for i in range(min(num_samples, 4)):  # Limit to first 4 samples to avoid too many files
                    pattern_name = dataset.pattern_identifiers[i] if i < len(dataset.pattern_identifiers) else f"sample_{i}"

                    # Create hidden states plot
                    cols_hidden = min(4, extra_channels)
                    rows_hidden = (extra_channels + cols_hidden - 1) // cols_hidden
                    fig_hidden, axs_hidden = plt.subplots(
                        rows_hidden, cols_hidden, figsize=(cols_hidden * 2.5, rows_hidden * 2.5)
                    )
                    fig_hidden.suptitle(f"Hidden States - {pattern_name}", fontsize=14)

                    if rows_hidden == 1:
                        axs_hidden = np.atleast_1d(axs_hidden)

                    # Get hidden channels from final result
                    hidden_channels = final_step[i, embedding_dim:].cpu().numpy()

                    for ch in range(extra_channels):
                        if rows_hidden > 1:
                            ax = axs_hidden[ch // cols_hidden, ch % cols_hidden]
                        else:
                            ax = axs_hidden[ch]

                        ax.imshow(hidden_channels[ch], cmap="viridis")
                        ax.axis("off")
                        ax.set_title(f"Hidden Ch {ch + 1}")

                    # Hide unused subplots
                    if rows_hidden > 1:
                        for ch in range(extra_channels, rows_hidden * cols_hidden):
                            axs_hidden[ch // cols_hidden, ch % cols_hidden].axis("off")

                    hidden_states_path = config["_path"] + f"/hidden_states_{pattern_name}_sample_{i}.png"
                    fig_hidden.savefig(hidden_states_path)
                    plt.close(fig_hidden)
                    print(f"Saved hidden states plot: {hidden_states_path}")

            # Create conditional growth grid visualization
            from src.utils.utils_plotting import plot_conditional_growth_grid

            plot_conditional_growth_grid(
                seed_images,
                target_images,
                result_images,
                dataset,
                sample_indices,
                config["id"],
                config["_path"],
                seed_type,
            )

            # Create growth animation showing NCA evolution
            id_ = config["id"]

            # Create dataset wrapper for animation
            class ConditionalGrowthDataset:
                def __init__(self, selected_patterns, samples):
                    self.pattern_identifiers = selected_patterns
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return (
                        self.samples[idx][0],  # initial_seed
                        self.samples[idx][1],  # task_encoder
                        self.samples[idx][2],  # target
                    )

            growth_dataset_for_animation = ConditionalGrowthDataset(
                dataset.pattern_identifiers,
                [(all_initial_seeds[i], all_task_encoders[i], all_targets[i]) for i in range(num_samples)],
            )

            create_animation_grid(
                growth_dataset_for_animation,
                embedding_dim,
                results.cpu().numpy(),
                output_path=animations_path,
                id_=f"{id_}_conditional_growth",
            )

            # Save conditional growth animation to wandb as video
            wandb.log(
                {
                    "conditional_growth_animation_train": wandb.Video(
                        f"{animations_path}/all_patterns_grid_anim_{id_}_conditional_growth.mp4"
                    )
                }
            )

            print(f"Conditional growth animation saved to {animations_path}/all_patterns_grid_anim_{id_}_conditional_growth.mp4")

            # Create evaluation dataset animations with pattern_list_eval
            if config["pattern_identifiers_eval"] is not None and config["pattern_identifiers_eval"]:
                print("Creating conditional growth animation for evaluation patterns...")
                # Create conditional growth dataset with evaluation patterns
                dataset_eval_patterns = GoalPatternsDataset(
                    size=pattern_size,
                    seed_type=seed_type,
                    embedding_dim=embedding_dim,
                    extra_channels=extra_channels,
                    one_hot_encoder=use_one_hot_encoder,
                    external_encoder_dim=len(config["pattern_identifiers_eval"]),
                    device=device,
                    dtype=dtype,
                    target_patterns=config["pattern_identifiers_eval"],
                    no_task_encoder=config["no_task_encoder"],
                    seed_positions=config["seed_positions"],
                )

                # Run evaluation on eval patterns
                num_eval_samples = len(dataset_eval_patterns)
                eval_sample_indices = list(range(num_eval_samples))

                # Batch eval samples
                eval_initial_seeds = []
                eval_task_encoders = []
                eval_targets = []
                for idx in eval_sample_indices:
                    initial_seed, task_encoder, target = dataset_eval_patterns[idx]
                    eval_initial_seeds.append(initial_seed)
                    eval_task_encoders.append(task_encoder)
                    eval_targets.append(target)

                eval_initial_seeds = torch.stack(eval_initial_seeds, dim=0).to(device)
                eval_targets = torch.stack(eval_targets, dim=0).to(device)

                # Handle task encoders - they might be None if no_task_encoder is True
                if config["no_task_encoder"]:
                    eval_task_encoders_processed = None
                else:
                    eval_task_encoders = torch.stack(eval_task_encoders, dim=0).to(device)
                    if external_encoder_layer is not None:
                        eval_task_encoders_processed = external_encoder_layer(eval_task_encoders)
                    else:
                        eval_task_encoders_processed = eval_task_encoders

                # Run evaluate_nca in batch for eval patterns
                with torch.no_grad():
                    eval_final_step, eval_results = evaluate_nca_batched(
                        model_to_load.to("cpu"),
                        eval_initial_seeds.cpu(),
                        (eval_task_encoders_processed.cpu() if eval_task_encoders_processed is not None else None),
                        nca_steps_eval,
                        additive=config["additive_update"],
                        state_norm=activation_state_norm,
                        update_noise=0.0,
                        alive_mask=config["alive_mask"],
                    )

                # Create MP4 animations for training results (like mNCA_evaluate_trained_model.py)
                print("Creating MP4 animations...")
                animations_dir = config["_path"] + "/animations/train"
                pathlib.Path(animations_dir).mkdir(parents=True, exist_ok=True)

                # eval_results shape: [time_steps, batch_size, channels, H, W]
                num_eval_samples = eval_results.shape[1]

                for pattern_idx in range(num_eval_samples):
                    pattern_name = (
                        config["pattern_identifiers_eval"][pattern_idx]
                        if pattern_idx < len(config["pattern_identifiers_eval"])
                        else f"sample_{pattern_idx}"
                    )

                    # Main pattern animation (embedding channels only)
                    states_to_viz = eval_results[:, pattern_idx, :embedding_dim, :, :].detach().cpu().numpy().clip(0, 1)

                    visualise_every_n_steps = 1
                    animation_path = f"{animations_dir}/nca_grown_{pattern_name}_pattern_{pattern_idx}_conditional_growth.mp4"
                    animate_states(
                        states_to_viz,
                        frames=range(0, states_to_viz.shape[0], visualise_every_n_steps),
                        interval=200,
                        cmap="gray" if embedding_dim == 1 else None,
                        filename=animation_path,
                    )
                    print(f"Saved animation: {animation_path}")

                    # Hidden channels animation (if there are extra channels)
                    nb_total_channels = embedding_dim + extra_channels
                    if nb_total_channels > embedding_dim:
                        hidden_states_to_viz = eval_results[:, pattern_idx, embedding_dim:, :, :].detach().cpu().numpy()

                        hidden_animation_path = (
                            f"{animations_dir}/hidden_channels_{pattern_name}_pattern_{pattern_idx}_conditional_growth.mp4"
                        )
                        animate_hidden_channels(
                            hidden_states_to_viz,
                            frames=range(0, hidden_states_to_viz.shape[0], visualise_every_n_steps),
                            interval=200,
                            filename=hidden_animation_path,
                        )
                        print(f"Saved hidden channels animation: {hidden_animation_path}")

                # Create dataset wrapper for animation
                class EvalConditionalGrowthDataset:
                    def __init__(self, selected_patterns, samples):
                        self.pattern_identifiers = selected_patterns
                        self.samples = samples

                    def __len__(self):
                        return len(self.samples)

                    def __getitem__(self, idx):
                        return (
                            self.samples[idx][0],  # initial_seed
                            self.samples[idx][1],  # task_encoder
                            self.samples[idx][2],  # target
                        )

                eval_growth_dataset_for_animation = EvalConditionalGrowthDataset(
                    config["pattern_identifiers_eval"],
                    [(eval_initial_seeds[i], eval_task_encoders[i], eval_targets[i]) for i in range(num_eval_samples)],
                )

                create_animation_grid(
                    eval_growth_dataset_for_animation,
                    embedding_dim,
                    eval_results.cpu().numpy(),
                    output_path=animations_eval_path,
                    id_=f"{config['id']}_eval_patterns_conditional_growth",
                )

                # Save evaluation conditional growth animation to wandb as video
                wandb.log(
                    {
                        "conditional_growth_animation_eval_patterns": wandb.Video(
                            f"{animations_eval_path}/all_patterns_grid_anim_{config['id']}_eval_patterns_conditional_growth.mp4"
                        )
                    }
                )

                print(
                    f"Evaluation conditional growth animation saved to {animations_eval_path}/all_patterns_grid_anim_{config['id']}_eval_patterns_conditional_growth.mp4"
                )

            # Create evaluation dataset animations with pattern_list_eval
            if config["pattern_identifiers_eval"] is not None:
                print("Creating trajectory animation for evaluation patterns...")
                # Create trajectory dataset with evaluation patterns

                dataset_eval_patterns = GoalPatternsTrajectoryDataset(
                    pattern_size=pattern_size,
                    space_size=config["space_size_eval"],
                    embedding_dim=embedding_dim,
                    extra_channels=extra_channels,
                    device=device,
                    dtype=dtype,
                    target_patterns=config["pattern_identifiers_eval"],
                    nca_steps=config["nca_steps"],  # Should be [min, max] for trajectory
                    boundary_condition=config["boundary_condition"],
                    num_samples_per_transformation=config["num_samples_per_transformation"],
                    domain_noise=0.0,  # No domain noise for evaluation
                )
                dataset_eval_patterns.eval()

                # Sample trajectory examples from eval dataset
                num_eval_samples = min(8, len(dataset_eval_patterns))
                eval_sample_indices = np.random.choice(len(dataset_eval_patterns), size=num_eval_samples, replace=False)

                # Collect samples for evaluation
                eval_input_images = []
                eval_task_encoders_sequences = []
                eval_target_images_sequences = []
                eval_seq_lengths = []

                for idx in eval_sample_indices:
                    input_img, task_enc_seq, target_img_seq = dataset_eval_patterns[idx]
                    eval_input_images.append(input_img)
                    eval_task_encoders_sequences.append(task_enc_seq)
                    eval_target_images_sequences.append(target_img_seq)
                    eval_seq_lengths.append(task_enc_seq.shape[0])

                # Use the trajectory collate function to handle variable lengths
                eval_batch = list(
                    zip(
                        eval_input_images,
                        eval_task_encoders_sequences,
                        eval_target_images_sequences,
                    )
                )
                (
                    eval_input_images,
                    eval_task_encoders_sequences,
                    eval_target_images_sequences,
                    eval_seq_lengths,
                ) = dataset_eval_patterns.trajectory_collate_fn(eval_batch)

                eval_input_images = eval_input_images.to(device)
                eval_task_encoders_sequences = eval_task_encoders_sequences.to(device)
                eval_target_images_sequences = eval_target_images_sequences.to(device)

                # Run NCA on eval trajectory samples
                with torch.no_grad():
                    batch_size_actual, num_channels, H, W = eval_input_images.shape
                    max_seq_len = eval_task_encoders_sequences.shape[1]

                    state = eval_input_images.clone()
                    eval_collected_states = []

                    # Run NCA for seq_len steps with different task encoders at each step
                    for step in range(max_seq_len):
                        # Get task encoder for current step
                        current_task_encoder = eval_task_encoders_sequences[:, step, :]  # [batch, 4]

                        # Transform task encoders through external encoder layer if it exists
                        if external_encoder_layer is not None:
                            task_encoders_processed = external_encoder_layer(current_task_encoder)
                        else:
                            task_encoders_processed = current_task_encoder

                        task_encoders_spatial = prepare_external_inputs(
                            task_encoders_processed, batch_size_actual, H
                        )

                        # Run single NCA step
                        state = nca_step(
                            state,
                            model_to_load,
                            task_encoders_spatial,
                            0.0,
                            config["additive_update"],
                            config["alive_mask"],
                            config["state_norm"],
                        )

                        # Collect state for this step
                        eval_collected_states.append(state[:, :embedding_dim])

                    # Convert collected_states to the format expected by create_animation_grid
                    eval_trajectory_states = torch.stack(eval_collected_states, dim=0)  # [seq_len, batch, embedding_dim, H, W]
                    eval_trajectory_states_np = eval_trajectory_states.cpu().numpy()

                    # Create a simple dataset wrapper for animation
                    class EvalTrajectoryDataset:
                        def __init__(self, dataset, sample_indices):
                            self.dataset = dataset
                            self.sample_indices = sample_indices
                            # Map sample indices to pattern names using the modulo operation
                            self.pattern_identifiers = [
                                dataset.pattern_identifiers[idx % dataset.num_patterns] for idx in sample_indices
                            ]

                        def __len__(self):
                            return len(self.sample_indices)

                        def __getitem__(self, idx):
                            # Return the original dataset item for the sampled index
                            return self.dataset[self.sample_indices[idx]]

                    eval_trajectory_dataset_wrapper = EvalTrajectoryDataset(dataset_eval_patterns, eval_sample_indices)

                    # Create animation grid for eval patterns
                    create_animation_grid(
                        eval_trajectory_dataset_wrapper,
                        embedding_dim,
                        eval_trajectory_states_np,
                        output_path=animations_eval_path,
                        id_=f"{config['id']}_eval_patterns_trajectory",
                    )

                    # Save evaluation trajectory animation to wandb as video
                    wandb.log(
                        {
                            "trajectory_animation_eval_patterns": wandb.Video(
                                f"{animations_eval_path}/all_patterns_grid_anim_{config['id']}_eval_patterns_trajectory.mp4"
                            )
                        }
                    )

                    print(
                        f"Evaluation trajectory animation saved to {animations_eval_path}/all_patterns_grid_anim_{config['id']}_eval_patterns_trajectory.mp4"
                    )

        print(f"Visualization saved to {config['_path']}")
