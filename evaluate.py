import yaml
import torch
import os
import argparse
from pathlib import Path
import time
import random
import numpy as np


from NCAs.NCA_mlp import NCA_mlp, evaluate_nca
from NCAs.utils import return_activation_function
from src.datasets.pattern_dataset import (
    GoalPatternsDataset,
    GoalPatternsTransformDataset,
    clear_pattern_cache,
)

import warnings

warnings.filterwarnings("ignore")


def _flush_stdin():
    """Discard any pending data in stdin buffer."""
    import sys, select, termios
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (termios.error, ValueError, OSError):
        pass


def _robust_input(prompt, retries=3):
    """input() wrapper that flushes stdin and retries on spurious SIGINT."""
    _flush_stdin()
    for attempt in range(retries):
        try:
            return input(prompt)
        except KeyboardInterrupt:
            _flush_stdin()
            if attempt < retries - 1:
                print(f"\n(Spurious interrupt caught, retrying... Ctrl+C {retries - 1 - attempt} more time(s) to exit)")
            else:
                raise


def load_emoji_from_unicode(pattern_char, size, embedding_dim, device, dtype):
    """Load and process an pattern from unicode character."""
    try:
        # Import emoji_to_numpy from the dataset module
        from src.datasets.pattern_dataset import emoji_to_numpy

        # Convert pattern to numpy array using the existing function
        pattern_np = emoji_to_numpy(pattern_char, size)

        # Convert to tensor
        img_tensor = torch.tensor(pattern_np, dtype=dtype, device=device)

        # Ensure we have the correct number of channels
        if img_tensor.shape[0] != embedding_dim:
            if embedding_dim == 3 and img_tensor.shape[0] == 4:
                # Convert RGBA to RGB by taking first 3 channels
                img_tensor = img_tensor[:3, :, :]
            elif embedding_dim == 4 and img_tensor.shape[0] == 3:
                # Convert RGB to RGBA by adding alpha channel
                alpha = torch.ones(1, size, size, dtype=dtype, device=device)
                img_tensor = torch.cat([img_tensor, alpha], dim=0)
            else:
                raise ValueError(f"Cannot convert {img_tensor.shape[0]} channels to {embedding_dim} channels")

        return img_tensor

    except Exception as e:
        print(f"Failed to load pattern '{pattern_char}': {e}")
        raise


def load_model_and_config(model_path):
    """Load saved model and its configuration."""
    config_path = os.path.join(model_path, "eval_config.yml")
    model_config_path = os.path.join(model_path, "model_config.yml")
    eval_config_path = os.path.join(model_path, "eval_config.yml")
    model_file = os.path.join(model_path, "best_model.pth")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config.yml for base parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model_config.yml for model architecture parameters
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        config.update(model_config)

    # Load eval_config.yml for evaluation/dataset parameters
    with open(eval_config_path, "r") as f:
        eval_config = yaml.safe_load(f)
    config.update(eval_config)
    print(f"Using eval_config.yml")
    use_precomputed = True

    # Map legacy or alternate task names to supported ones
    if config["task"] == "patterns_translation_trajectory":
        config["task"] = "patterns_translation"

    # Default transformation_amount if missing from eval_config
    if "transformation_amount" not in config:
        config["transformation_amount"] = 1

    # Set up device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["dtype"] == "float64":
        dtype = torch.float64
    elif config["dtype"] == "float32":
        dtype = torch.float32
    elif config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)

    # Parse activation functions
    activation_conv = return_activation_function(config["activation_conv"])
    activation_fc = return_activation_function(config["activation_fc"])
    activation_last = return_activation_function(config["activation_last"])

    # Robust handling for state_norm: true/false in config
    state_norm_value = config["state_norm"]
    if state_norm_value is True or (isinstance(state_norm_value, str) and state_norm_value.lower() == "true"):
        state_norm_value = "clamp(0,1)"
    elif state_norm_value is False or (isinstance(state_norm_value, str) and state_norm_value.lower() == "false"):
        state_norm_value = "identity"
    activation_state_norm = return_activation_function(state_norm_value)

    # Parse encoder activation if available
    activation_encoder = None
    if "activation_encoder" in config:
        activation_encoder = return_activation_function(config["activation_encoder"])

    # Use pre-computed dimensions if available (from eval_config.yml)
    if use_precomputed and "total_external_channels" in config:
        # eval_config.yml has all the pre-computed dimensions we need
        nca_external_channels = config["total_external_channels"]
        one_hot_dim = config["one_hot_dim"]
        num_position_channels = config.get("num_position_channels", 0)
        has_external_encoder = config.get("has_external_encoder", False)

        print(f"Using pre-computed dimensions:")
        print(f"  - one_hot_dim: {one_hot_dim}")
        print(f"  - num_position_channels: {num_position_channels}")
        print(f"  - total_external_channels: {nca_external_channels}")
        print(f"  - has_external_encoder: {has_external_encoder}")
    else:
        raise NotImplementedError("Dynamic dimension inference not available for this run.")

    # Create external encoder layer if it was used during training
    external_encoder_layer = None
    external_encoder_file = os.path.join(model_path, "best_external_encoder.pth")

    if has_external_encoder and os.path.exists(external_encoder_file):
        if activation_encoder is None:
            activation_encoder = torch.nn.Tanh()

        external_encoder_dim = config.get("external_encoder_dim", nca_external_channels - num_position_channels)

        external_encoder_layer = torch.nn.Sequential(
            torch.nn.Linear(one_hot_dim, external_encoder_dim, bias=config["bias"]),
            activation_encoder,
        ).to(device)

        external_encoder_layer.load_state_dict(torch.load(external_encoder_file, map_location=device))
        external_encoder_layer.eval()

        print(f"Loaded external encoder layer: {one_hot_dim} -> {external_encoder_dim}")


    # Determine num_output_conv_features based on convolution mode
    convolution_mode = config["convolution_mode"]
    if convolution_mode in ["share_kernels_across_channels", "one_kernel_per_channel"]:
        num_output_conv_features = None
    elif convolution_mode == "mixing_features":
        num_output_conv_features = config["num_output_conv_features"]
    else:
        raise ValueError(f"Unknown convolution_mode: {convolution_mode}")

    # Create model using config
    model = NCA_mlp(
        num_input_channels=config["embedding_dim"] + config["extra_channels"],
        num_external_channels=nca_external_channels,
        num_output_conv_features=num_output_conv_features,
        num_conv_layers=config["num_conv_layers"],
        hidden_dim_mlp=config["hidden_dim_mlp"],
        bias=config["bias"],
        activation_conv=activation_conv,
        activation_fc=activation_fc,
        activation_last=activation_last,
        stochastic_update_ratio=config["stochastic_update_ratio"],
        convolution_mode=convolution_mode,
        fixed_kernels=config["fixed_kernels"],
        num_kernels=config["num_kernels"],
        custom_kernels=config["custom_kernels"],
        width_kernel=config["width_kernel"],
        additive_update=config["additive_update"],
        merge_ext=config["merge_ext"],
        dropout=0.0,
        alive_mask_goal=config["alive_mask_goal"],
        alive_threshold=config["alive_threshold"],
        boundary_condition=config["boundary_condition"],
        isotropic_only=config["isotropic_only"],
        extra_kernels=config["extra_kernels"],
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Handle nca_steps as tuple/list for evaluation - use consistent middle value
    nca_steps = config["nca_steps"]
    if isinstance(nca_steps, (tuple, list)) and len(nca_steps) == 2:
        nca_steps_min, nca_steps_max = nca_steps
        nca_steps_eval = (nca_steps_min + nca_steps_max) // 2
        config["nca_steps_eval"] = nca_steps_eval
        config["nca_steps_is_range"] = True
        print(f"NCA steps loaded as range [{nca_steps_min}, {nca_steps_max}], using {nca_steps_eval} for evaluation")
    else:
        config["nca_steps_eval"] = nca_steps
        config["nca_steps_is_range"] = False
        print(f"NCA steps loaded as fixed value: {nca_steps}")

    return (
        model,
        config,
        device,
        dtype,
        external_encoder_layer,
        one_hot_dim,
        activation_state_norm,
    )


def create_interpolation_vector(selected_indices, one_hot_dim):
    """Create interpolation vector from selected indices."""
    task_vector = torch.zeros(one_hot_dim, dtype=torch.float32)
    if len(selected_indices) > 0:
        weight_per_index = 1.0 / len(selected_indices)
        for idx in selected_indices:
            task_vector[idx] = weight_per_index
    return task_vector


def get_task_vector_input(task_type, one_hot_dim, pattern_identifiers=None):
    """Get task vector input from user based on task type."""
    print(f"\n=== Task Vector Input for {task_type} ===")

    if task_type == "patterns_morphing":
        # For morphing, user selects the TARGET pattern to morph to
        if pattern_identifiers is not None:
            print("\nAvailable patterns:")
            for i, pattern in enumerate(pattern_identifiers):
                print(f"  {i}: {pattern}")
        print(f"Select TARGET pattern to morph to (0 to {one_hot_dim - 1})")

        while True:
            try:
                input_str = _robust_input("Enter target pattern index: ")
                pattern_index = int(input_str.strip())

                if pattern_index < 0 or pattern_index >= one_hot_dim:
                    print(f"Error: Index must be between 0 and {one_hot_dim - 1}")
                    continue

                # Create one-hot vector for target
                task_vector = torch.zeros(one_hot_dim, dtype=torch.float32)
                task_vector[pattern_index] = 1.0

                print(f"Task vector: {task_vector.numpy()}")
                return task_vector

            except ValueError:
                print("Error: Please enter a valid integer")
            except Exception as e:
                print(f"Error parsing input: {e}")

    elif task_type == "patterns_conditional_growth":
        # Ask user to choose between single pattern and interpolation mode
        print("\nChoose input mode:")
        print("1. Single pattern mode (one-hot encoding)")
        print("2. Interpolation mode (multiple patterns)")

        while True:
            mode_input = _robust_input("Enter mode (1 or 2): ").strip()
            if mode_input == "1":
                mode = "single"
                break
            elif mode_input == "2":
                mode = "interpolation"
                break
            else:
                print("Please enter 1 or 2")

        if mode == "single":
            print(f"\nEnter pattern index for generation (0 to {one_hot_dim - 1})")

            while True:
                try:
                    input_str = _robust_input("Enter pattern index: ")
                    pattern_index = int(input_str.strip())

                    if pattern_index < 0 or pattern_index >= one_hot_dim:
                        print(f"Error: Index must be between 0 and {one_hot_dim - 1}")
                        continue

                    # Create one-hot vector
                    task_vector = torch.zeros(one_hot_dim, dtype=torch.float32)
                    task_vector[pattern_index] = 1.0

                    print(f"Task vector: {task_vector.numpy()}")
                    return task_vector

                except ValueError:
                    print("Error: Please enter a valid integer")
                except Exception as e:
                    print(f"Error parsing input: {e}")

        else:  # interpolation mode
            print(f"Enter pattern indices for interpolation (0 to {one_hot_dim - 1})")
            print("Press Enter with no input when done selecting indices")

            selected_indices = []
            while True:
                try:
                    input_str = _robust_input(f"Enter pattern index {len(selected_indices) + 1} (or Enter to finish): ").strip()

                    # If empty input, user is done selecting
                    if not input_str:
                        if len(selected_indices) == 0:
                            print("Error: At least one index must be selected")
                            continue
                        break

                    pattern_index = int(input_str)

                    if pattern_index < 0 or pattern_index >= one_hot_dim:
                        print(f"Error: Index must be between 0 and {one_hot_dim - 1}")
                        continue

                    if pattern_index in selected_indices:
                        print(f"Error: Index {pattern_index} already selected")
                        continue

                    selected_indices.append(pattern_index)
                    print(f"Selected indices so far: {selected_indices}")

                except ValueError:
                    print("Error: Please enter a valid integer")
                except Exception as e:
                    print(f"Error parsing input: {e}")

            # Create interpolation vector
            task_vector = torch.zeros(one_hot_dim, dtype=torch.float32)
            weight_per_index = 1.0 / len(selected_indices)

            for idx in selected_indices:
                task_vector[idx] = weight_per_index

            print(f"Selected indices: {selected_indices}")
            print(f"Interpolation vector: {task_vector.numpy()}")
            return task_vector

    elif task_type == "patterns_rotation":
        print("Enter task vector for pattern transformation (2 dimensions)")
        print("Examples:")
        print("  [1, 0] - Rotate clockwise")
        print("  [0, 1] - Rotate anti-clockwise")
        print("  [0, 0] - No change")

    elif task_type == "patterns_translation":
        TRANSLATION_DIRS = {
            "0": ("up",    [1, 0, 0, 0]),
            "1": ("right", [0, 1, 0, 0]),
            "2": ("down",  [0, 0, 1, 0]),
            "3": ("left",  [0, 0, 0, 1]),
            "4": ("stay",  [0, 0, 0, 0]),
        }
        print("Enter direction index or a sequence of indices:")
        for k, (name, vec) in TRANSLATION_DIRS.items():
            print(f"  {k} - {name:5s}  {vec}")
        print("Examples:")
        print("  0         - single move up (then press C to repeat)")
        print("  0,1,2,3   - sequence: up, right, down, left")

        while True:
            input_str = _robust_input("Direction(s): ").strip()
            try:
                tokens = [t.strip() for t in input_str.split(",")]
                # Check if all tokens are valid direction indices
                if all(t in TRANSLATION_DIRS for t in tokens):
                    vectors = [torch.tensor(TRANSLATION_DIRS[t][1], dtype=torch.float32) for t in tokens]
                    names = [TRANSLATION_DIRS[t][0] for t in tokens]
                    if len(vectors) == 1:
                        print(f"Direction: {names[0]} {vectors[0].numpy()}")
                        return vectors[0]
                    else:
                        print(f"Sequence: {' -> '.join(names)}")
                        return vectors
                else:
                    # Fall back to raw vector parsing
                    if input_str.startswith("[") and input_str.endswith("]"):
                        input_str = input_str[1:-1]
                    values = [float(x.strip()) for x in input_str.split(",")]
                    if len(values) != one_hot_dim:
                        print(f"Error: Expected {one_hot_dim} values or direction indices (0-4), got: {input_str}")
                        continue
                    task_vector = torch.tensor(values, dtype=torch.float32)
                    print(f"Task vector: {task_vector.numpy()}")
                    return task_vector
            except Exception as e:
                print(f"Error: {e}. Enter indices (0-4) or a full vector.")

    while True:
        try:
            input_str = _robust_input("Enter task vector as comma-separated values: ")
            # Parse the input
            if input_str.startswith("[") and input_str.endswith("]"):
                input_str = input_str[1:-1]  # Remove brackets

            values = [float(x.strip()) for x in input_str.split(",")]

            if len(values) != one_hot_dim:
                print(f"Error: Expected {one_hot_dim} values, got {len(values)}")
                continue

            task_vector = torch.tensor(values, dtype=torch.float32)
            print(f"Task vector: {task_vector.numpy()}")
            return task_vector

        except Exception as e:
            print(f"Error parsing input: {e}")
            print("Please enter values as comma-separated numbers (e.g., 1, 0, 0)")


def get_initial_state(config, device, dtype):
    """Get or create initial state based on task type."""
    if config["task"] == "patterns_morphing":
        # For morphing tasks, initial state is a FULL source pattern (not a single cell)
        size = config["space_size"]
        embedding_dim = config["embedding_dim"]
        extra_channels = config["extra_channels"]

        # Load pattern dataset to get source patterns
        dataset = GoalPatternsDataset(
            size=config["pattern_size"],
            seed_type="zeros",  # Doesn't matter, we only use .targets directly
            embedding_dim=embedding_dim,
            extra_channels=extra_channels,
            one_hot_encoder=config["use_one_hot_encoder"],
            external_encoder_dim=config.get("one_hot_dim", len(config["target_patterns"])),
            device=device,
            dtype=dtype,
            target_patterns=config["target_patterns"],
        )

        # Show available patterns and let user select source pattern
        print("\nAvailable patterns for morphing (select SOURCE pattern):")
        for i, pattern in enumerate(dataset.pattern_identifiers):
            print(f"  {i}: {pattern}")

        while True:
            try:
                user_input = _robust_input(f"Select source pattern index (0-{len(dataset.pattern_identifiers)-1}): ").strip()

                if user_input.isdigit():
                    pattern_idx = int(user_input)
                    if 0 <= pattern_idx < len(dataset.pattern_identifiers):
                        break
                    print(f"Please enter a number between 0 and {len(dataset.pattern_identifiers)-1}")
                else:
                    print("Please enter a valid number")
            except ValueError:
                print("Please enter a valid number")

        source_pattern = dataset.targets[pattern_idx]

        # Create seed with full source pattern
        seed = torch.zeros(embedding_dim + extra_channels, size, size, device=device, dtype=dtype)

        # Place source pattern in seed
        if config["pattern_size"] == config["space_size"]:
            seed[:embedding_dim, :, :] = source_pattern
        else:
            center = size // 2
            pattern_size = config["pattern_size"]
            start_pos = center - pattern_size // 2
            end_pos = start_pos + pattern_size
            seed[:embedding_dim, start_pos:end_pos, start_pos:end_pos] = source_pattern

        if extra_channels > 0:
            seed[embedding_dim:, :, :] = 0.0

        print(f"Using source pattern: {dataset.pattern_identifiers[pattern_idx]}")
        return seed.unsqueeze(0), dataset.pattern_identifiers

    elif config["task"] == "patterns_conditional_growth":
        # Create seed for pattern generation
        size = config["space_size"]
        embedding_dim = config["embedding_dim"]
        extra_channels = config["extra_channels"]
        seed_type = config["seed_type"]

        seed = torch.zeros(embedding_dim + extra_channels, size, size, device=device, dtype=dtype)

        if seed_type == "single_cell_ones_all":
            seed[:, size // 2, size // 2] = 1.0
        elif seed_type == "single_cell_RGB_ones_OG":
            seed[3:, size // 2, size // 2] = 1.0
        elif seed_type == "single_cell_RGB_ones_OG_reversed":
            seed[:3, size // 2, size // 2] = 1.0
        elif seed_type == "double_cell_RGB_ones_all":
            seed[:3, size // 2 - 1 : size // 2 + 1, size // 2 - 1 : size // 2 + 1] = 1.0
        elif seed_type == "double_cell_RGB_ones_OG":
            seed[3:, size // 2 - 1 : size // 2 + 1, size // 2 - 1 : size // 2 + 1] = 1.0
        elif seed_type == "double_cell_RGB_ones_OG_reversed":
            seed[:3, size // 2 - 1 : size // 2 + 1, size // 2 - 1 : size // 2 + 1] = 1.0
        elif seed_type == "single_cell_RGB_random":
            seed[:3, size // 2, size // 2] = torch.rand(4, device=device, dtype=dtype) * 2 - 1
        elif seed_type == "single_cell_random":
            seed[:, size // 2, size // 2] = torch.rand(embedding_dim + extra_channels, device=device, dtype=dtype) * 2 - 1
        elif seed_type == "single_cell_ones":
            seed[:, size // 2, size // 2] = 1.0
        elif seed_type == "all_cells_random":
            seed = torch.rand(embedding_dim + extra_channels, size, size, device=device, dtype=dtype) * 2 - 1
        elif seed_type == "all_cells_ones":
            seed = torch.ones(embedding_dim + extra_channels, size, size, device=device, dtype=dtype)
        else:
            raise ValueError(f"Invalid seed type: {seed_type}")

        return seed.unsqueeze(0)  # Add batch dimension

    elif config["task"] == "patterns_rotation":
        # Load pattern dataset to get actual pattern images
        print("\nAvailable patterns for transformation:")
        dataset = GoalPatternsTransformDataset(
            pattern_size=config["pattern_size"],
            space_size=config["space_size"],
            embedding_dim=config["embedding_dim"],
            extra_channels=config["extra_channels"],
            device=device,
            dtype=dtype,
            target_patterns=config["target_patterns"],  # Use pattern list from config (required)
            transformation_amount=config["transformation_amount"],
            transformation_type="rotation",
            boundary_condition=config["boundary_condition"],
            num_samples_per_transformation=1,
            domain_noise=0.0,
            batch_size=1,
        )

        # Show available patterns
        for i, pattern in enumerate(dataset.pattern_identifiers):
            print(f"  {i}: {pattern}")
        print(f"  Or type any pattern character (e.g., 🧽, 🚗, 🍕)")

        while True:
            try:
                user_input = _robust_input(f"Select pattern index (0-{len(dataset.pattern_identifiers)-1}) or type an pattern: ").strip()

                # Check if input is a number (existing pattern)
                if user_input.isdigit():
                    pattern_idx = int(user_input)
                    if 0 <= pattern_idx < len(dataset.pattern_identifiers):
                        break
                    print(f"Please enter a number between 0 and {len(dataset.pattern_identifiers)-1}")
                else:
                    # Handle direct pattern input
                    if len(user_input) > 0:
                        try:
                            # Load pattern from unicode character
                            custom_pattern_tensor = load_emoji_from_unicode(
                                user_input,
                                config["pattern_size"],
                                config["embedding_dim"],
                                device,
                                dtype,
                            )

                            # Embed custom pattern in the center of the space
                            center = config["space_size"] // 2
                            embedded = dataset._embed_pattern_in_space(custom_pattern_tensor, center, center)

                            # Add extra channels if needed
                            if config["extra_channels"] > 0:
                                extra_padding = torch.zeros(
                                    config["extra_channels"],
                                    config["space_size"],
                                    config["space_size"],
                                    dtype=dtype,
                                    device=device,
                                )
                                embedded = torch.cat([embedded, extra_padding], dim=0)

                            print(f"Custom pattern '{user_input}' loaded successfully!")
                            print(f"Custom pattern tensor shape: {custom_pattern_tensor.shape}")
                            print(f"Embedded shape: {embedded.shape}")
                            print(f"Pattern size: {config['pattern_size']}, Space size: {config['space_size']}")

                            return embedded.unsqueeze(0), f"custom_pattern_{user_input}"

                        except Exception as e:
                            print(f"Error loading pattern '{user_input}': {e}")
                            retry = _robust_input("Try again? (y/n): ").strip().lower()
                            if retry != "y":
                                continue
                    else:
                        print("Please enter a valid pattern or number")
            except ValueError:
                print("Please enter a valid number or pattern character")

        # Get the pattern tensor (this is pattern_size x pattern_size)
        pattern_tensor = dataset.pattern_tensors[pattern_idx]

        # Embed pattern in the center of the space (this creates space_size x space_size)
        center = config["space_size"] // 2
        embedded = dataset._embed_pattern_in_space(pattern_tensor, center, center)

        # Add extra channels if needed
        if config["extra_channels"] > 0:
            extra_padding = torch.zeros(
                config["extra_channels"],
                config["space_size"],
                config["space_size"],
                dtype=dtype,
                device=device,
            )
            embedded = torch.cat([embedded, extra_padding], dim=0)

        print(f"Pattern tensor shape: {pattern_tensor.shape}")
        print(f"Embedded shape: {embedded.shape}")
        print(f"Pattern size: {config['pattern_size']}, Space size: {config['space_size']}")

        return (
            embedded.unsqueeze(0),
            dataset.pattern_identifiers[pattern_idx],
        )  # Add batch dimension

    elif config["task"] == "patterns_translation":
        # Load pattern dataset to get actual pattern images
        print("\nAvailable patterns for translation:")
        dataset = GoalPatternsTransformDataset(
            pattern_size=config["pattern_size"],
            space_size=config["space_size"],
            embedding_dim=config["embedding_dim"],
            extra_channels=config["extra_channels"],
            device=device,
            dtype=dtype,
            target_patterns=config["target_patterns"],
            transformation_amount=config["transformation_amount"],
            transformation_type="translation",
            boundary_condition=config["boundary_condition"],
            num_samples_per_transformation=1,
            domain_noise=0.0,
            batch_size=1,
        )

        # Show available patterns
        for i, pattern in enumerate(dataset.pattern_identifiers):
            print(f"  {i}: {pattern}")
        print(f"  Or type any pattern character (e.g., 🧽, 🚗, 🍕)")

        while True:
            try:
                user_input = _robust_input(f"Select pattern index (0-{len(dataset.pattern_identifiers)-1}) or type an pattern: ").strip()

                # Check if input is a number (existing pattern)
                if user_input.isdigit():
                    pattern_idx = int(user_input)
                    if 0 <= pattern_idx < len(dataset.pattern_identifiers):
                        break
                    print(f"Please enter a number between 0 and {len(dataset.pattern_identifiers)-1}")
                else:
                    # Handle direct pattern input
                    if len(user_input) > 0:
                        try:
                            # Load pattern from unicode character
                            custom_pattern_tensor = load_emoji_from_unicode(
                                user_input,
                                config["pattern_size"],
                                config["embedding_dim"],
                                device,
                                dtype,
                            )

                            # Embed custom pattern in the center of the space
                            center = config["space_size"] // 2
                            embedded = dataset._embed_pattern_in_space(custom_pattern_tensor, center, center)

                            # Add extra channels if needed
                            if config["extra_channels"] > 0:
                                extra_padding = torch.zeros(
                                    config["extra_channels"],
                                    config["space_size"],
                                    config["space_size"],
                                    dtype=dtype,
                                    device=device,
                                )
                                embedded = torch.cat([embedded, extra_padding], dim=0)

                            print(f"Custom pattern '{user_input}' loaded successfully!")
                            print(f"Custom pattern tensor shape: {custom_pattern_tensor.shape}")
                            print(f"Embedded shape: {embedded.shape}")
                            print(f"Pattern size: {config['pattern_size']}, Space size: {config['space_size']}")

                            return embedded.unsqueeze(0), f"custom_pattern_{user_input}"

                        except Exception as e:
                            print(f"Error loading pattern '{user_input}': {e}")
                            retry = _robust_input("Try again? (y/n): ").strip().lower()
                            if retry != "y":
                                continue
                    else:
                        print("Please enter a valid pattern or number")
            except ValueError:
                print("Please enter a valid number or pattern character")

        # Get the pattern tensor (this is pattern_size x pattern_size)
        pattern_tensor = dataset.pattern_tensors[pattern_idx]

        # Embed pattern in the center of the space (this creates space_size x space_size)
        center = config["space_size"] // 2
        embedded = dataset._embed_pattern_in_space(pattern_tensor, center, center)

        # Add extra channels if needed
        if config["extra_channels"] > 0:
            extra_padding = torch.zeros(
                config["extra_channels"],
                config["space_size"],
                config["space_size"],
                dtype=dtype,
                device=device,
            )
            embedded = torch.cat([embedded, extra_padding], dim=0)

        print(f"Pattern tensor shape: {pattern_tensor.shape}")
        print(f"Embedded shape: {embedded.shape}")
        print(f"Pattern size: {config['pattern_size']}, Space size: {config['space_size']}")

        return (
            embedded.unsqueeze(0),
            dataset.pattern_identifiers[pattern_idx],
        )  # Add batch dimension


# def run_evaluation(model, initial_state, task_vector, config, activation_state_norm):
#     """Run the NCA evaluation using the existing evaluate_nca function."""
#     nca_steps = config["nca_steps_eval"]

#     if config["nca_steps_is_range"]:
#         original_range = config["nca_steps"]
#         print(f"\nRunning NCA for {nca_steps} steps (middle of range {original_range})...")
#     else:
#         print(f"\nRunning NCA for {nca_steps} steps...")

#     # The existing evaluate_nca expects single samples (no batch dimension)
#     # and separate initial_state and task_vector parameters
#     initial_state_single = initial_state.squeeze(0)  # Remove batch dimension

#     # Use the existing evaluate_nca function from NCA_mlp.py
#     final_step, results = evaluate_nca(
#         model=model,
#         initial_state=initial_state_single,
#         encoder_vector=task_vector,
#         nca_steps=nca_steps,
#         additive=config["additive_update"],
#         state_norm=activation_state_norm,
#         alive_mask=config["alive_mask"],
#         update_noise=0.0,
#     )

#     return final_step, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NCA models")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to saved model directory (if not provided, will prompt)",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear the pattern cache before running evaluation",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (headless mode, saves final frame as PNG instead)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        help="Translation direction(s) to skip interactive prompt (e.g. '0' or '0,1,2,3')",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern index to skip interactive prompt (e.g. '0')",
    )
    parser.add_argument(
        "--space-size",
        type=int,
        default=None,
        help="Override canvas size for evaluation (e.g. 128). Defaults to training space_size.",
    )

    args = parser.parse_args()

    if args.no_gui:
        import matplotlib
        matplotlib.use("Agg")

    # Handle cache clearing
    if args.clear_cache:
        clear_pattern_cache()
        print("Pattern cache cleared. Exiting.")
        return 0

    # Print cache info
    cache_dir = os.path.expanduser("~/.cache/goalNCA/patterns")
    if os.path.exists(cache_dir):
        cache_files = len([f for f in os.listdir(cache_dir) if f.endswith(".pkl")])
        print(f"📦 Using pattern cache: {cache_files} cached patterns in {cache_dir}")
        print("   (Use --clear_cache to clear cache if needed)")
    else:
        print("📦 Pattern cache will be created on first run for faster subsequent evaluations")
    print()

    # Get model path
    if args.model_path:
        model_path = args.model_path
    else:
        print("Available models in results/:")
        results_dir = Path("results")
        available_models = {}  # Store mapping of index -> full_path
        model_index = 0

        if results_dir.exists():
            for task_dir in results_dir.iterdir():
                if task_dir.is_dir():
                    print(f"\n{task_dir.name}:")
                    for model_dir in task_dir.iterdir():
                        if model_dir.is_dir() and (model_dir / "eval_config.yml").exists():
                            print(f"  [{model_index}] {model_dir.name}")
                            available_models[str(model_index)] = str(model_dir)
                            model_index += 1

        user_input = _robust_input(f"\nEnter model index (0-{model_index-1}) or full path: ")

        # Check if user entered an index number vs full path
        if user_input in available_models:
            model_path = available_models[user_input]
            print(f"Using model: {model_path}")
        else:
            model_path = user_input

    try:
        # Load model and config
        print(f"Loading model from {model_path}...")
        (
            model,
            config,
            device,
            dtype,
            external_encoder_layer,
            one_hot_dim,
            activation_state_norm,
        ) = load_model_and_config(model_path)
        print(f"Model loaded successfully!")
        print(f"Task: {config['task']}")
        print(f"Device: {device}")
        if external_encoder_layer is not None:
            print(f"External encoder layer loaded: {one_hot_dim} -> {config['external_encoder_dim']}")

        if args.space_size is not None:
            print(f"Overriding space_size: {config['space_size']} -> {args.space_size}")
            config["space_size"] = args.space_size
        space_size_eval = config["space_size"]
        # Standard evaluation mode
        while True:
            try:
                # Get initial state and task vector
                # Order depends on task type for better UX
                if config["task"] == "patterns_morphing":
                    initial_state, pattern_identifiers = get_initial_state(config, device, dtype)
                    task_vector = get_task_vector_input(config["task"], one_hot_dim, pattern_identifiers)
                    pattern_name = None
                elif config["task"] == "patterns_rotation":
                    task_vector = get_task_vector_input(config["task"], one_hot_dim)
                    initial_state, pattern_name = get_initial_state(config, device, dtype)
                elif config["task"] == "patterns_translation":
                    if args.direction is not None:
                        # Parse --direction arg same as interactive input
                        TRANSLATION_DIRS = {
                            "0": [1,0,0,0], "1": [0,1,0,0],
                            "2": [0,0,1,0], "3": [0,0,0,1], "4": [0,0,0,0],
                        }
                        tokens = [t.strip() for t in args.direction.split(",")]
                        vectors = [torch.tensor(TRANSLATION_DIRS[t], dtype=torch.float32) for t in tokens]
                        task_vector = vectors[0] if len(vectors) == 1 else vectors
                    else:
                        task_vector = get_task_vector_input(config["task"], one_hot_dim)
                    if args.pattern is not None:
                        # Skip interactive pattern selection
                        dataset = GoalPatternsTransformDataset(
                            pattern_size=config["pattern_size"],
                            space_size=config["space_size"],
                            embedding_dim=config["embedding_dim"],
                            extra_channels=config["extra_channels"],
                            device=device, dtype=dtype,
                            target_patterns=config["target_patterns"],
                            transformation_amount=config["transformation_amount"],
                            transformation_type="translation",
                            boundary_condition=config["boundary_condition"],
                            num_samples_per_transformation=1,
                            domain_noise=0.0, batch_size=1,
                        )
                        pattern_idx = int(args.pattern)
                        pattern_tensor = dataset.pattern_tensors[pattern_idx]
                        center = config["space_size"] // 2
                        embedded = dataset._embed_pattern_in_space(pattern_tensor, center, center)
                        if config["extra_channels"] > 0:
                            extra_padding = torch.zeros(config["extra_channels"], config["space_size"], config["space_size"], dtype=dtype, device=device)
                            embedded = torch.cat([embedded, extra_padding], dim=0)
                        initial_state = embedded.unsqueeze(0)
                        pattern_name = dataset.pattern_identifiers[pattern_idx]
                    else:
                        initial_state, pattern_name = get_initial_state(config, device, dtype)
                elif config["task"] == "patterns_conditional_growth":
                    task_vector = get_task_vector_input(config["task"], one_hot_dim)
                    initial_state = get_initial_state(config, device, dtype)
                    pattern_name = None
                else:
                    raise ValueError(f"Unknown task type: {config['task']}")

                # Run live animation
                try:
                    config["alive_threshold"] = model.alive_threshold
                    from src.visualisation.visualisation_functions import run_live_animation

                    def _encode_vector(tv):
                        if external_encoder_layer is not None:
                            with torch.no_grad():
                                return external_encoder_layer(tv.unsqueeze(0)).squeeze(0)
                        return tv

                    # Build output dir: evaluations/{experiment_name}/
                    experiment_name = os.path.basename(os.path.normpath(model_path))
                    eval_output_dir = os.path.join("evaluations", experiment_name)

                    # Handle sequence of task vectors (e.g. translation sequence)
                    headless = args.no_gui
                    if isinstance(task_vector, list):
                        task_vectors_processed = [_encode_vector(tv) for tv in task_vector]
                        run_live_animation(
                            config,
                            model,
                            initial_state,
                            task_vectors_processed,
                            device,
                            pattern_name,
                            activation_state_norm,
                            headless=headless,
                            output_dir=eval_output_dir,
                        )
                    else:
                        task_vector_processed = _encode_vector(task_vector)
                        run_live_animation(
                            config,
                            model,
                            initial_state,
                            task_vector_processed,
                            device,
                            pattern_name,
                            activation_state_norm,
                            headless=headless,
                            output_dir=eval_output_dir,
                        )
                except KeyboardInterrupt:
                    print("\n⏹️  Live animation interrupted by user")
                except Exception as e:
                    print(f"\nError in live animation: {e}")

                if args.no_gui:
                    break
                # Ask if user wants to continue
                continue_eval = _robust_input("\nDo you want to try another evaluation? (y/n): ").lower()
                if continue_eval != "y" and continue_eval != "yes":
                    break

            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user")
                break
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if args.no_gui:
                    break
                continue_eval = _robust_input("Do you want to try again? (y/n): ").lower()
                if continue_eval != "y" and continue_eval != "yes":
                    break

    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    print("Evaluation completed!")
    return 0


if __name__ == "__main__":
    exit(main())
