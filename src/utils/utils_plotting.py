import numpy as np
import matplotlib.pyplot as plt
import math
import wandb


def plot_training_curve(all_losses, all_learning_rates, config_id, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale plot
    line1 = ax1.plot(
        all_losses,
        linewidth=1,
        alpha=0.8,
        color="blue",
        label=f"Loss (Min: {all_losses.min():.6f})",
    )
    ax1.set_title(f"Training Loss & Learning Rate (Linear) - {config_id}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)

    # Add secondary y-axis for learning rate
    ax1_lr = ax1.twinx()
    line2 = ax1_lr.plot(
        all_learning_rates, linewidth=1, alpha=0.8, color="red", label=f"Learning Rate"
    )
    ax1_lr.set_ylabel("Learning Rate", color="red")
    ax1_lr.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Log scale plot
    line3 = ax2.plot(
        all_losses,
        linewidth=1,
        alpha=0.8,
        color="blue",
        label=f"Loss (Min: {all_losses.min():.6f})",
    )
    ax2.set_title(f"Training Loss & Learning Rate (Log) - {config_id}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)", color="blue")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.grid(True, alpha=0.3)

    # Add secondary y-axis for learning rate
    ax2_lr = ax2.twinx()
    line4 = ax2_lr.plot(
        all_learning_rates, linewidth=1, alpha=0.8, color="red", label=f"Learning Rate"
    )
    ax2_lr.set_ylabel("Learning Rate", color="red")
    ax2_lr.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{output_path}/training_curve_{config_id}.pdf", bbox_inches="tight")
    wandb.log({"training_curve": wandb.Image(plt)})
    plt.close()


def plot_checksum_l1_curve(all_l1_checksums, all_l1_checksums_eval, config_id, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale plot
    if len(all_l1_checksums) > 0:
        train_epochs = np.arange(0, len(all_l1_checksums) * 100, 100)
        ax1.plot(
            train_epochs,
            all_l1_checksums,
            linewidth=1.5,
            alpha=0.8,
            color="green",
            label=f"Train Checksum L1 (Min: {all_l1_checksums.min():.6f})",
            marker="o",
            markersize=3,
        )

    if len(all_l1_checksums_eval) > 0 and all_l1_checksums_eval[0] is not None:
        eval_epochs = np.arange(0, len(all_l1_checksums_eval) * 100, 100)
        ax1.plot(
            eval_epochs,
            all_l1_checksums_eval,
            linewidth=1.5,
            alpha=0.8,
            color="orange",
            label=f"Eval Checksum L1 (Min: {all_l1_checksums_eval.min():.6f})",
            marker="s",
            markersize=3,
        )

    ax1.set_title(f"Checksum L1 Distance (Linear) - {config_id}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Checksum L1 Distance")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Log scale plot
    if len(all_l1_checksums) > 0:
        ax2.plot(
            train_epochs,
            all_l1_checksums,
            linewidth=1.5,
            alpha=0.8,
            color="green",
            label=f"Train Checksum L1 (Min: {all_l1_checksums.min():.6f})",
            marker="o",
            markersize=3,
        )

    if len(all_l1_checksums_eval) > 0 and all_l1_checksums_eval[0] is not None:
        ax2.plot(
            eval_epochs,
            all_l1_checksums_eval,
            linewidth=1.5,
            alpha=0.8,
            color="orange",
            label=f"Eval Checksum L1 (Min: {all_l1_checksums_eval.min():.6f})",
            marker="s",
            markersize=3,
        )

    ax2.set_title(f"Checksum L1 Distance (Log) - {config_id}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Checksum L1 Distance (log scale)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{output_path}/checksum_l1_curve_{config_id}.pdf", bbox_inches="tight")
    wandb.log({"checksum_l1_curve": wandb.Image(plt)})
    plt.close()


def plot_morphing_grid(
    source_images, target_images, result_images, dataset, sample_indices, config_id, output_path
):
    num_patterns = len(dataset.pattern_identifiers)
    grid_rows = num_patterns
    grid_cols = num_patterns
    total_cells = grid_rows * grid_cols

    # Pad with empty images if needed
    if len(source_images) > 0:
        empty_img = np.ones_like(source_images[0])
        while len(source_images) < total_cells:
            source_images.append(empty_img)
            target_images.append(empty_img)
            result_images.append(empty_img)

    # Plot organized grid: each row = source emoji, each col = target emoji
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 3))
    if grid_rows == 1 and grid_cols == 1:
        axs = np.array([[axs]])
    elif grid_rows == 1:
        axs = axs.reshape((1, grid_cols))
    elif grid_cols == 1:
        axs = axs.reshape((grid_rows, 1))

    num_samples = len(sample_indices)
    for i in range(num_samples):
        row = i // grid_cols
        col = i % grid_cols

        # Concatenate source, target, and result horizontally
        cell_img = np.hstack([source_images[i], target_images[i], result_images[i]])
        sample_info = dataset.get_sample_info(sample_indices[i])

        # Enhanced title with source and target info
        # Support both 'source_emoji'/'target_emoji' (emojis) and 'source_pattern'/'target_pattern' (patterns)
        source = sample_info.get('source_emoji') or sample_info.get('source_pattern', '')
        target = sample_info.get('target_emoji') or sample_info.get('target_pattern', '')

        if col == 0:
            title = f"From {source}\n→ {target}"
        else:
            title = f"→ {target}"

        axs[row, col].set_title(title)
        axs[row, col].imshow(cell_img)
        axs[row, col].axis("off")

    # Hide any unused subplots
    for i in range(num_samples, total_cells):
        row = i // grid_cols
        col = i % grid_cols
        axs[row, col].axis("off")
    plt.tight_layout()

    wandb.log({"morphing_combinations_grid": wandb.Image(plt)})
    fig.savefig(f"{output_path}/morphing_combinations_grid_{config_id}.png")
    plt.close(fig)


def plot_transformations_grid(
    input_images,
    target_images,
    result_images,
    dataset,
    sample_indices,
    config_id,
    output_path,
    task_type,
):
    num_samples = len(sample_indices)
    grid_size = math.ceil(math.sqrt(num_samples))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 3))
    if grid_size == 1:
        axs = np.array([[axs]])
    elif len(axs.shape) == 1:
        axs = axs.reshape((1, grid_size))

    for i in range(num_samples):
        row = i // grid_size
        col = i % grid_size

        # Get sample info
        sample_info = dataset.get_sample_info(sample_indices[i])

        # Create visualization: input | target | result
        combined_img = np.hstack([input_images[i], target_images[i], result_images[i]])
        axs[row, col].imshow(combined_img)

        if task_type == "rotation":
            title = f"{sample_info['transformation']}\n{sample_info['input_angle']}°→{sample_info['target_angle']}°"
        elif task_type == "translation":
            title = f"{sample_info['transformation']}\n({sample_info['center_x']},{sample_info['center_y']})→({sample_info['target_center_x']},{sample_info['target_center_y']})"
        else:
            title = f"{sample_info['transformation']}"

        axs[row, col].set_title(title)
        axs[row, col].axis("off")

    # Hide unused subplots
    for i in range(num_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axs[row, col].axis("off")

    plt.tight_layout()

    wandb_key = f"{task_type}_transformations_grid"
    wandb.log({wandb_key: wandb.Image(plt)})

    filename = (
        f"transformations_grid_{config_id}.png"
        if task_type == "rotation"
        else f"translations_grid_{config_id}.png"
    )
    fig.savefig(f"{output_path}/{filename}")
    plt.close(fig)


def plot_conditional_growth_grid(
    seed_images,
    target_images,
    result_images,
    dataset,
    sample_indices,
    config_id,
    output_path,
    seed_type,
):
    """Plot conditional growth results showing seed -> target -> result for each emoji."""
    num_samples = len(sample_indices)
    grid_size = math.ceil(math.sqrt(num_samples))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 3))
    if grid_size == 1:
        axs = np.array([[axs]])
    elif len(axs.shape) == 1:
        axs = axs.reshape((1, grid_size))

    for i in range(num_samples):
        row = i // grid_size
        col = i % grid_size

        # Create visualization: seed | target | result
        combined_img = np.hstack([seed_images[i], target_images[i], result_images[i]])
        axs[row, col].imshow(combined_img)

        # Get emoji name from dataset
        emoji_name = dataset.pattern_identifiers[i]
        title = f"{emoji_name}\nSeed → Target → Result"

        axs[row, col].set_title(title, fontsize=10)
        axs[row, col].axis("off")

    # Hide unused subplots
    for i in range(num_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axs[row, col].axis("off")

    plt.suptitle(f"Conditional Growth Results - {seed_type} seeds", fontsize=14)
    plt.tight_layout()

    wandb.log({"conditional_growth_grid": wandb.Image(plt)})

    filename = f"conditional_growth_grid_{config_id}.png"
    fig.savefig(f"{output_path}/{filename}")
    plt.close(fig)


def plot_gradient_magnitudes(all_gradient_magnitudes, config_id, output_path):
    """
    Plot gradient magnitudes over training epochs.

    Args:
        all_gradient_magnitudes: numpy array of gradient magnitudes per epoch
        config_id: run identifier for naming
        output_path: directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale plot
    ax1.plot(
        all_gradient_magnitudes,
        linewidth=1,
        alpha=0.8,
        color="purple",
        label=f"Gradient Magnitude (Min: {all_gradient_magnitudes.min():.6f}, Max: {all_gradient_magnitudes.max():.6f})",
    )
    ax1.set_title(f"Gradient Magnitudes (Linear) - {config_id}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gradient Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Log scale plot
    ax2.plot(
        all_gradient_magnitudes,
        linewidth=1,
        alpha=0.8,
        color="purple",
        label=f"Gradient Magnitude (Min: {all_gradient_magnitudes.min():.6f}, Max: {all_gradient_magnitudes.max():.6f})",
    )
    ax2.set_title(f"Gradient Magnitudes (Log) - {config_id}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Magnitude (log scale)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{output_path}/gradient_magnitudes_{config_id}.pdf", bbox_inches="tight")
    wandb.log({"gradient_magnitudes_curve": wandb.Image(plt)})
    plt.close()


def plot_perturbation_losses(losses, smoothness_weight, output_path):
    """
    Plot perturbation optimization losses with both linear and log scales.

    Args:
        losses: dict with keys "total", "mse", "smoothness", optionally "smoothness_scaled"
        smoothness_weight: float indicating if smoothness was used
        output_path: full path to save the plot (e.g., "results/loss_curve.pdf")
    """
    if smoothness_weight > 0.0:
        fig, axes = plt.subplots(3, 2, figsize=(16, 8))

        axes[0, 0].plot(losses["total"])
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("Linear Scale")
        axes[0, 0].grid(True)

        axes[0, 1].plot(losses["total"])
        axes[0, 1].set_ylabel("Total Loss")
        axes[0, 1].set_title("Log Scale")
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True)

        axes[1, 0].plot(losses["mse"])
        axes[1, 0].set_ylabel("MSE Loss")
        axes[1, 0].grid(True)

        axes[1, 1].plot(losses["mse"])
        axes[1, 1].set_ylabel("MSE Loss")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True)

        axes[2, 0].plot(losses["smoothness"])
        axes[2, 0].set_ylabel("Smoothness Loss (raw)")
        axes[2, 0].set_xlabel("Iteration")
        axes[2, 0].grid(True)

        axes[2, 1].plot(losses["smoothness"])
        axes[2, 1].set_ylabel("Smoothness Loss (raw)")
        axes[2, 1].set_xlabel("Iteration")
        axes[2, 1].set_yscale("log")
        axes[2, 1].grid(True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        axes[0].plot(losses["mse"])
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_xlabel("Iteration")
        axes[0].set_title("Linear Scale")
        axes[0].grid(True)

        axes[1].plot(losses["mse"])
        axes[1].set_ylabel("MSE Loss")
        axes[1].set_xlabel("Iteration")
        axes[1].set_title("Log Scale")
        axes[1].set_yscale("log")
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_nca_growth_video(states_list, output_path):
    """
    Save NCA growth animation using collected states.

    Args:
        states_list: List of [C, H, W] numpy arrays (RGBA states at each step)
        output_path: Path to save MP4 file
    """
    from src.visualisation.viz import animate_states

    states_array = np.stack(states_list, axis=0)

    animate_states(
        states_array,
        frames=None,
        interval=100,
        filename=output_path,
        cmap=None,
    )

    print(f"  - nca_growth.mp4 ({len(states_list)} frames)")


def plot_target_output_comparison(target_np, final_output_np, output_path):
    """
    Create side-by-side comparison plot of target and final output.

    Args:
        target_np: numpy array [H, W, C] with values in [0, 1]
        final_output_np: numpy array [H, W, C] with values in [0, 1]
        output_path: path to save the comparison plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target_np)
    axes[0].set_title("Target")
    axes[0].axis("off")
    axes[1].imshow(final_output_np)
    axes[1].set_title("Final Output")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
