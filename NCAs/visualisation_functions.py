import os
import subprocess
import matplotlib.pyplot as plt
import time
from celluloid import Camera
import numpy as np
import imageio
import io
from tqdm import tqdm
from matplotlib import animation
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import imageio
import os
import time
import matplotlib.pyplot as plt
import io
import h5py
import math
import torch

from NCAs.utils import return_activation_function, compute_alive_mask


def process_frame(state, color, label, cmap, quality):
    if color and state.shape[0] >= 3:
        frame = np.transpose(state[:3, :, :], (1, 2, 0))
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
    else:
        color_map = plt.cm.get_cmap(cmap)
        frame = (255 * color_map(state[0])).astype(np.uint8)

    with io.BytesIO() as buf:
        imageio.imwrite(buf, frame, format="png", compress_level=quality)
        buf.seek(0)
        return imageio.v2.imread(buf)


def create_animation_imageio_parallel(states, quality, output_path, color, label, cmap, max_cores=16):
    print(f"Creating animation with states of shape {states.shape}...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fps = int(len(states) / 10)
    timestamp = str(time.time()).split(".")[0]
    video_path = f"{output_path}/animation_imageio_seed_{str(label)}.mp4"

    # Use format-specific writer options
    writer_options = {
        "fps": fps,
        "macro_block_size": None,
    }

    # Only add ffmpeg_params if we're writing to MP4 format
    if video_path.endswith(".mp4"):
        writer_options["ffmpeg_params"] = [
            "-probesize",
            "200M",
            "-framerate",
            str(fps),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ]

    frames = []
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = [executor.submit(process_frame, state, color, label, cmap, quality) for state in states]
        for future in futures:
            frames.append(future.result())

    # Explicitly specify format to ensure MP4 writer is used
    with imageio.get_writer(video_path, format="FFMPEG", **writer_options) as writer:
        for frame in frames:
            writer.append_data(frame)


def create_animation_celluloid(states, quality, output_path, color, label, cmap):
    # Create figure and adjust its settings
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Initialize Camera
    camera = Camera(fig)
    tic = time.time()

    # Capture frames
    print("\nCreating animation...")
    for i in tqdm(range(len(states)), desc="Creating animation"):
        if color and states[i].shape[0] >= 3:
            s = np.transpose(states[i][:3, :, :], (1, 2, 0))
            ax.imshow(s)
        else:
            s = states[i]
            ax.imshow(s, cmap=cmap, vmin=0, vmax=1)

        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        camera.snap()

    # Create and save the animation
    animation = camera.animate(interval=1)
    timestamp = str(time.time()).split(".")[0]
    output_path_ = os.path.join(output_path, f"animation_celluloid_seed_{label}.mp4")
    # extra_args = ["-vcodec", "libx264", "-crf", str(quality)]
    extra_args = ["-crf", str(quality)]
    animation.save(output_path_, fps=int(len(states) / 10), extra_args=extra_args)


def create_animation_ffmpeg(states, quality, output_path, color, label, cmap):
    # Ensure the output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Calculate fps and construct the output file path
    fps = int(len(states) / 10)
    timestamp = str(time.time()).split(".")[0]
    output_path = os.path.join(output_path, f"animation_ffmpeg_seed_{label}.mp4")

    # Start ffmpeg process
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vcodec",
        "libx264",
        "-crf",
        str(quality),
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Process and send each frame to ffmpeg
    for state in states:
        if color and state.shape[0] >= 3:
            frame = np.transpose(state[:3, :, :], (1, 2, 0))
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = state
            frame_cmap = plt.cm.get_cmap(cmap)
            frame_colormap = frame_cmap(frame)  # Apply colormap
            plt.imsave(ffmpeg_process.stdin, frame_colormap, format="png")

    # Close ffmpeg process
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    if ffmpeg_process.returncode != 0:
        stderr = ffmpeg_process.stderr.read().decode()
        print(f"ffmpeg error: {stderr}")
        raise subprocess.CalledProcessError(ffmpeg_process.returncode, ffmpeg_cmd)


def plot_v_int_distribution(labels):
    plt.figure()
    plt.hist(labels, bins=50)
    plt.xlabel("v_int value")
    plt.ylabel("Frequency")
    # plt.show()


def plot_weight_distributions(weights, labels, percentiles):
    # Convert weights and labels to numpy arrays for easier manipulation
    weights = np.array(weights)
    labels = np.array(labels)

    # Calculate the percentiles of labels
    percentile_values = np.percentile(labels, [100 - p for p in percentiles])

    # Prepare the plot
    num_plots = len(percentiles) + 1  # +1 to include the histogram for all weights
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5), sharey=True)
    fig.suptitle("Weight Distributions by Label Percentiles")

    # Plot for all weights
    axs[0].hist(weights.flatten(), bins=20, color="grey", alpha=0.7)
    axs[0].set_title("All Weights")
    axs[0].set_xlabel("Weight Value")
    axs[0].set_ylabel("Frequency")

    # Plot for each percentile
    for i, percentile in enumerate(percentiles):
        # Find indexes of labels within the current percentile
        indexes = labels <= percentile_values[i]

        # Select weights for the current percentile
        current_weights = weights[indexes]

        # Plot
        axs[i + 1].hist(current_weights.flatten(), bins=20, alpha=0.7)
        axs[i + 1].set_title(f"Top {percentile}%")
        axs[i + 1].set_xlabel("Weight Value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()


def visualize_3d_growth(states_to_viz, path=None):
    """
    Visualize 3D voxel data interactively with time progression for a batch of structures.
    If input is 5D (frames, channels, x, y, z), renders a single structure.
    The animation can restart by pressing the spacebar.

    Parameters:
    - states_to_viz: Tensor of shape (batch_size, frames, channels, x, y, z) or (frames, channels, x, y, z)
    - path: Optional path to save the animation as an mp4 file.
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    # Check if input is 5D or 6D
    if len(states_to_viz.shape) == 5:
        states_to_viz = states_to_viz[None, ...]  # Add batch dimension
    batch_size = states_to_viz.shape[0]

    fig, axes = plt.subplots(1, batch_size, subplot_kw={"projection": "3d"}, figsize=(5 * batch_size, 5))

    if batch_size == 1:
        axes = [axes]  # Ensure axes is always a list

    # Set consistent limits for all axes
    for ax in axes:
        ax.set_xlim([0, states_to_viz.shape[3]])
        ax.set_ylim([0, states_to_viz.shape[4]])
        ax.set_zlim([0, states_to_viz.shape[5]])

    # Initialize plot with the first frame for each batch
    def plot_frame(frame):
        for b in range(batch_size):
            ax = axes[b]
            data = states_to_viz[b, frame]  # shape (channels, x, y, z)

            # Compute voxels: active where any channel is non-zero
            voxels = np.any(data > 0, axis=0)  # shape (x, y, z)

            # Extract colors from the channels
            num_channels = data.shape[0]
            if num_channels == 4:
                colors = np.transpose(data, (1, 2, 3, 0))  # RGBA
            elif num_channels == 3:
                alpha_channel = np.ones(data.shape[1:], dtype=data.dtype)
                data_with_alpha = np.concatenate([data, alpha_channel[None, ...]], axis=0)
                colors = np.transpose(data_with_alpha, (1, 2, 3, 0))
            else:
                raise ValueError("Data must have 3 (RGB) or 4 (RGBA) channels.")

            colors = np.clip(colors, 0, 1)
            colors[~voxels] = [0, 0, 0, 0]  # Transparent non-active voxels

            # Plot the voxel grid
            ax.voxels(voxels, facecolors=colors, edgecolor="k")
            ax.set_title(f"Batch {b}, Frame {frame}")

    def update():
        for frame in range(states_to_viz.shape[1]):
            for ax in axes:
                ax.clear()  # Clear previous frame
                ax.set_xlim([0, states_to_viz.shape[3]])
                ax.set_ylim([0, states_to_viz.shape[4]])
                ax.set_zlim([0, states_to_viz.shape[5]])
            plot_frame(frame)
            plt.pause(0.1)  # Pause between frames for animation effect

    def on_key(event):
        if event.key == " ":  # Spacebar to restart the animation
            update()

    # Connect the key event
    fig.canvas.mpl_connect("key_press_event", on_key)

    if path is not None:
        from matplotlib.animation import FuncAnimation

        def animate(frame):
            for ax in axes:
                ax.clear()
                ax.set_xlim([0, states_to_viz.shape[3]])
                ax.set_ylim([0, states_to_viz.shape[4]])
                ax.set_zlim([0, states_to_viz.shape[5]])
            plot_frame(frame)

        ani = FuncAnimation(fig, animate, frames=states_to_viz.shape[1], interval=100)
        ani.save(path, writer="ffmpeg")
    else:
        # Start the animation interactively
        update()


def render_voxel_structure(voxel_batch, path=None):
    # Ensure voxel_batch is at least 4D (add batch dimension if it's a single grid)
    if voxel_batch.shape[0] == 3:  # No batch dimension
        voxel_batch = voxel_batch[None, ...]

    batch_size = voxel_batch.shape[0]
    fig, axs = plt.subplots(1, batch_size, subplot_kw={"projection": "3d"}, figsize=(batch_size * 5, 5))

    if batch_size == 1:
        axs = [axs]  # Ensure axs is always iterable

    for i, voxel_grid in enumerate(voxel_batch):
        ax = axs[i]

        # Extract RGB channels
        r, g, b = voxel_grid

        # Create color array from the RGB channels
        colors = np.stack((r, g, b), axis=-1)

        # Get the coordinates of filled voxels
        filled = np.any(voxel_grid > 0, axis=0)

        # Plot the voxels with the corresponding colors
        ax.voxels(filled, facecolors=colors, edgecolors="k", alpha=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    if path:
        plt.savefig(path)
    else:
        plt.show()


def visualize_all_patterns_grid(dataset, embedding_dim, results, id_=None):
    """
    Visualize all patterns in the dataset in a square grid.
    Each cell shows target (top) and NCA (bottom) for each emoji.
    'results' must be a numpy array of shape (batch_size, channels, H, W), containing the final state for each emoji.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    num_patterns = len(dataset)
    grid_size = math.ceil(math.sqrt(num_patterns))
    total_cells = grid_size**2

    # Prepare arrays for targets and results
    target_images = []
    result_images = []

    for idx in range(num_patterns):
        _, _, target = dataset[idx]
        target = target[:embedding_dim].cpu().numpy().transpose(1, 2, 0)
        result = results[idx, :embedding_dim].clip(0, 1).transpose(1, 2, 0)
        # Always concatenate alpha channel of zeros
        if target.shape[-1] == 3:
            alpha = np.zeros(target.shape[:2], dtype=target.dtype)
            target = np.concatenate([target, alpha[..., None]], axis=-1)
        if result.shape[-1] == 3:
            alpha = np.zeros(result.shape[:2], dtype=result.dtype)
            result = np.concatenate([result, alpha[..., None]], axis=-1)
        target_images.append(target)
        result_images.append(result)

    # Pad with empty images if needed
    empty_img = np.ones_like(target_images[0])
    while len(target_images) < total_cells:
        target_images.append(empty_img)
        result_images.append(empty_img)

    # Plot grid: each cell shows target (top) and NCA (bottom)
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2.5))
    for i in range(total_cells):
        row = i // grid_size
        col = i % grid_size
        # Stack target and NCA vertically for each cell
        cell_img = np.vstack([target_images[i], result_images[i]])
        axs[row, col].imshow(cell_img)
        if i < num_patterns:
            axs[row, col].set_title(f"{i+1}")
        axs[row, col].axis("off")
    plt.tight_layout()
    if id_ is None:
        id_ = np.random.randint(0, 10e6)
    fig.savefig(f"media/all_patterns_grid_{id_}.png")
    plt.close(fig)


def create_animation_grid(dataset, embedding_dim, results, output_path, id_=None, cmap=None):
    """
    Create a grid animation where each cell shows the NCA evolution for one emoji in the dataset.
    - dataset: the dataset object (for targets and labels)
    - embedding_dim: number of channels to visualize (e.g. 3 for RGB)
    - results: numpy array of shape (n_steps+1, batch, channels, H, W)
    - output_path: directory to save the animation
    - id_: optional identifier for the output file
    - cmap: optional colormap for grayscale
    The animation is saved as an mp4 file in output_path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    import math
    import os

    n_steps, batch, channels, H, W = results.shape
    num_patterns = len(dataset)
    grid_size = math.ceil(math.sqrt(num_patterns))
    total_cells = grid_size**2

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2.5))
    if grid_size == 1:
        axs = np.array([[axs]])
    elif len(axs.shape) == 1:
        axs = axs.reshape((grid_size, grid_size))

    ims = []
    for i in range(total_cells):
        row = i // grid_size
        col = i % grid_size
        ax = axs[row, col]
        im = ax.imshow(np.ones((H, W, embedding_dim)), animated=True)
        ax.axis("off")
        ims.append(im)

    def update(frame):
        for i in range(total_cells):
            row = i // grid_size
            col = i % grid_size
            ax = axs[row, col]
            if i < num_patterns:
                img = results[frame, i, :embedding_dim].clip(0, 1).transpose(1, 2, 0)
                # Always concatenate alpha channel of zeros
                # if img.shape[-1] == 3: #! HACK
                #     alpha = np.zeros(img.shape[:2], dtype=img.dtype)
                #     img = np.concatenate([img, alpha[..., None]], axis=-1)
                # if embedding_dim == 1:
                #     ims[i].set_data(img[:, :, 0])
                #     ims[i].set_cmap(cmap)
                #     ims[i].set_clim(0, 1)
                # else:
                #     ims[i].set_data(img)
                ims[i].set_data(img)
                ax.set_title(f"{i+1}")
            else:
                ims[i].set_data(np.ones((H, W, embedding_dim)))
                ax.set_title("")
        return ims

    anim = FuncAnimation(fig, update, frames=n_steps, blit=False, interval=100)
    if id_ is None:
        id_ = np.random.randint(0, 10e6)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    anim.save(
        os.path.join(output_path, f"all_patterns_grid_anim_{id_}.mp4"),
        fps=10,
        extra_args=["-crf", "23"],
    )
    plt.close(fig)


def create_transform_animation(
    config,
    model,
    dataset,
    embedding_dim,
    nca_steps,
    output_path,
    device="cpu",
    id_=None,
    external_encoder_layer=None,
):
    """
    Create a transformation animation for the emoji transformation task.
    Handles both rotation and translation animations based on the dataset transformation_type.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    import math
    import os
    import torch

    print(f"Creating {dataset.transformation_type} animation...")

    # Set up the model
    model.eval()
    model.to(device)

    # Get unique patterns from dataset
    unique_patterns = []
    for emoji in dataset.pattern_identifiers:
        if emoji not in [e[0] for e in unique_patterns]:
            # Find the emoji tensor
            for i, sample_emoji in enumerate(dataset.pattern_identifiers):
                if sample_emoji == emoji:
                    emoji_tensor = dataset.pattern_tensors[i]
                    unique_patterns.append((emoji, emoji_tensor))
                    break

    num_patterns = len(unique_patterns)
    grid_size = math.ceil(math.sqrt(num_patterns))

    if dataset.transformation_type == "rotation":
        # Create rotation animation
        # Create clockwise rotation task encoder
        clockwise_encoder = torch.tensor([1.0, 0.0], dtype=dataset.dtype, device=device)

        # Collect all rotation states for each emoji
        all_transform_states = []

        for emoji_name, emoji_tensor in unique_patterns:
            print(f"Processing {emoji_name} rotation...")

            # Start with emoji centered in the larger space
            center_x = dataset.space_size // 2
            center_y = dataset.space_size // 2

            # Embed emoji in space (this creates a space_size x space_size tensor)
            initial_state = dataset._embed_pattern_in_space(emoji_tensor, center_x, center_y)

            if dataset.extra_channels > 0:
                extra_padding = torch.zeros(
                    dataset.extra_channels,
                    dataset.space_size,
                    dataset.space_size,
                    dtype=dataset.dtype,
                    device=device,
                )
                initial_state = torch.cat([initial_state, extra_padding], dim=0)

            # Add batch dimension
            initial_state = initial_state.unsqueeze(0)  # [1, C, H, W]
            clockwise_encoder_batch = clockwise_encoder.unsqueeze(0)  # [1, 2]

            # Collect states for full rotation (360 degrees)
            transform_states = []
            current_state = initial_state.clone()
            transform_states.append(current_state)

            # Transform encoder through external layer if it exists
            if external_encoder_layer is not None:
                clockwise_encoder_processed = external_encoder_layer(clockwise_encoder_batch)
            else:
                clockwise_encoder_processed = clockwise_encoder_batch

            combined_external = (
                clockwise_encoder_processed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dataset.space_size, dataset.space_size)
            )

            # Apply rotation 360 times (one for each degree)
            for degree in range(360):
                # Run NCA for nca_steps to get rotation
                with torch.no_grad():
                    for _ in range(nca_steps):
                        if config["alive_mask"]:
                            pre_life_mask = compute_alive_mask(current_state, model.alive_threshold)
                        if model.additive_update:
                            current_state = current_state + model(current_state, x_ext=combined_external, update_noise=0.0)
                        else:
                            current_state = model(current_state, x_ext=combined_external, update_noise=0.0)
                        if config["alive_mask"]:
                            post_life_mask = compute_alive_mask(current_state, model.alive_threshold)
                            life_mask = (pre_life_mask & post_life_mask).float()
                            current_state = current_state * life_mask
                        if config["state_norm"]:
                            current_state = torch.clamp(current_state, 0, 1)
                transform_states.append(current_state.clone())

            # Convert to numpy array and store
            transform_states = torch.stack(transform_states, dim=0)  # [361, 1, C, H, W]
            transform_states = transform_states.squeeze(1)  # [361, C, H, W]
            transform_states = transform_states.cpu().numpy()
            all_transform_states.append(transform_states)

        # Stack all emoji rotations
        all_transform_states = np.stack(all_transform_states, axis=1)  # [361, num_patterns, C, H, W]
        num_frames = 361
        animation_name = "rotation"

    elif dataset.transformation_type == "translation":
        # Create translation animation - generate 5 separate videos
        # Create translation task encoders (including identity transformation)
        translation_encoders = [
            ("up", torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dataset.dtype, device=device)),
            ("right", torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=dataset.dtype, device=device)),
            ("down", torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=dataset.dtype, device=device)),
            ("left", torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dataset.dtype, device=device)),
            ("no_movement", torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=dataset.dtype, device=device)),
        ]

        # Generate 5 separate videos
        output_files = []

        for direction_name, direction_encoder in translation_encoders:
            print(f"Creating {direction_name} animation...")

            # Collect all translation states for each emoji with this specific task vector
            all_transform_states = []

            for emoji_name, emoji_tensor in unique_patterns:
                print(f"  Processing {emoji_name} for {direction_name}...")

                # Start with emoji centered in the larger space
                center_x = dataset.space_size // 2
                center_y = dataset.space_size // 2

                # Embed emoji in space (this creates a space_size x space_size tensor)
                initial_state = dataset._embed_pattern_in_space(emoji_tensor, center_x, center_y)

                if dataset.extra_channels > 0:
                    extra_padding = torch.zeros(
                        dataset.extra_channels,
                        dataset.space_size,
                        dataset.space_size,
                        dtype=dataset.dtype,
                        device=device,
                    )
                    initial_state = torch.cat([initial_state, extra_padding], dim=0)

                # Add batch dimension
                initial_state = initial_state.unsqueeze(0)  # [1, C, H, W]
                direction_encoder_batch = direction_encoder.unsqueeze(0)  # [1, 4]

                # Create a sequence showing repeated application of this task vector
                transform_states = []
                current_state = initial_state.clone()

                # Transform direction encoder through external encoder layer if it exists
                if external_encoder_layer is not None:
                    direction_encoder_processed = external_encoder_layer(direction_encoder_batch)
                else:
                    direction_encoder_processed = direction_encoder_batch

                combined_external = (
                    direction_encoder_processed.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, dataset.space_size, dataset.space_size)
                )

                # Apply the same transformation repeatedly
                num_steps = 50  # Show 50 steps of the same transformation
                for step in range(num_steps):
                    transform_states.append(current_state.clone())

                    # Apply transformation
                    with torch.no_grad():
                        for _ in range(nca_steps):
                            if config["alive_mask"]:
                                pre_life_mask = compute_alive_mask(current_state, model.alive_threshold)
                            if model.additive_update:
                                current_state = current_state + model(current_state, x_ext=combined_external, update_noise=0.0)
                            else:
                                current_state = model(current_state, x_ext=combined_external, update_noise=0.0)
                            if config["alive_mask"]:
                                post_life_mask = compute_alive_mask(current_state, model.alive_threshold)
                                life_mask = (pre_life_mask & post_life_mask).float()
                                current_state = current_state * life_mask
                            if config["state_norm"]:
                                current_state = torch.clamp(current_state, 0, 1)

                # Convert to numpy array and store
                transform_states = torch.stack(transform_states, dim=0)  # [num_steps, 1, C, H, W]
                transform_states = transform_states.squeeze(1)  # [num_steps, C, H, W]
                transform_states = transform_states.cpu().numpy()
                all_transform_states.append(transform_states)

            # Stack all emoji translations for this direction
            all_transform_states = np.stack(all_transform_states, axis=1)  # [num_steps, num_patterns, C, H, W]

            # Create the animation for this specific direction
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
            if grid_size == 1:
                axs = np.array([[axs]])
            elif len(axs.shape) == 1:
                axs = axs.reshape((grid_size, grid_size))

            # Initialize images
            ims = []
            for i in range(grid_size * grid_size):
                row = i // grid_size
                col = i % grid_size
                ax = axs[row, col]

                if i < num_patterns:
                    # Show initial frame
                    img = all_transform_states[0, i, :embedding_dim].clip(0, 1).transpose(1, 2, 0)
                    im = ax.imshow(img, animated=True)
                    ax.set_title(f"{unique_patterns[i][0]} - {direction_name}")
                else:
                    # Empty subplot
                    im = ax.imshow(
                        np.ones((dataset.space_size, dataset.space_size, embedding_dim)),
                        animated=True,
                    )
                    ax.set_title("")

                ax.axis("off")
                ims.append(im)

            def update_direction(frame):
                for i in range(grid_size * grid_size):
                    if i < num_patterns:
                        img = all_transform_states[frame, i, :embedding_dim].clip(0, 1).transpose(1, 2, 0)
                        ims[i].set_data(img)
                        axs[i // grid_size, i % grid_size].set_title(
                            f"{unique_patterns[i][0]} - {direction_name} (step {frame+1})"
                        )
                return ims

            # Create animation for this direction
            anim = FuncAnimation(fig, update_direction, frames=num_steps, blit=False, interval=100)

            # Save animation
            if id_ is None:
                id_ = np.random.randint(0, 10e6)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_file = os.path.join(output_path, f"translation_{direction_name}_animation_{id_}.mp4")
            anim.save(output_file, fps=10, extra_args=["-crf", "18"])

            plt.close(fig)
            output_files.append(output_file)
            print(f"  {direction_name} animation saved to {output_file}")

        print(f"All translation animations saved: {len(output_files)} files")
        return output_files  # Return list of files instead of single file

    else:
        raise ValueError(f"Unknown transformation_type: {dataset.transformation_type}")

    print(f"Animation data shape: {all_transform_states.shape}")

    # Create the animation
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    if grid_size == 1:
        axs = np.array([[axs]])
    elif len(axs.shape) == 1:
        axs = axs.reshape((grid_size, grid_size))

    # Initialize images
    ims = []
    for i in range(grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        ax = axs[row, col]

        if i < num_patterns:
            # Show initial frame
            img = all_transform_states[0, i, :embedding_dim].clip(0, 1).transpose(1, 2, 0)
            im = ax.imshow(img, animated=True)
            ax.set_title(f"{unique_patterns[i][0]}")
        else:
            # Empty subplot
            im = ax.imshow(np.ones((dataset.space_size, dataset.space_size, embedding_dim)), animated=True)
            ax.set_title("")

        ax.axis("off")
        ims.append(im)

    def update(frame):
        for i in range(grid_size * grid_size):
            if i < num_patterns:
                img = all_transform_states[frame, i, :embedding_dim].clip(0, 1).transpose(1, 2, 0)
                ims[i].set_data(img)

                # Update title with current transformation info
                degree = frame
                axs[i // grid_size, i % grid_size].set_title(f"{unique_patterns[i][0]} ({degree}°)")
        return ims

    # Create animation
    interval = 50
    anim = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=interval)

    # Save animation
    if id_ is None:
        id_ = np.random.randint(0, 10e6)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, f"{animation_name}_animation_{id_}.mp4")
    fps = 20
    anim.save(output_file, fps=fps, extra_args=["-crf", "18"])

    plt.close(fig)
    print(f"{animation_name.capitalize()} animation saved to {output_file}")

    return [output_file]  # Return as list for consistency


if __name__ == "__main__":
    import h5py

    # Load the compressed file
    seed = 2623070694
    file = "arrays/states_to_viz_" + str(seed) + ".h5"
    # file = "arrays/states_to_viz_" + str(seed) + "_compressed.npz"
    print(f"\nLoading array from {file}...")
    tic = time.time()
    with h5py.File(file, "r") as file:
        states_to_viz = file["array"][:]
    # states_to_viz = np.load(file)["array"]
    print(f"Loading array took: {time.time() - tic:.2f} seconds\n\n")
    print("Creating animation...")
    tic = time.time()
    create_animation_imageio_parallel(
        states_to_viz,
        quality=0,
        output_path="media",
        color=True,
        label="_torch_" + str(seed),
        cmap="Spectral",
    )  # quality range 0-9, 0 is no compression which is the fastest
    # create_animation_imageio(states_to_viz, quality=0, output_path="media", color=True, label="_torch_" + str(seed), cmap="Spectral")  # quality range 0-9, 0 is no compression which is the fastest
    # create_animation_celluloid(states, quality=23, output_path="media", color=True, cmap="Spectral")  # quality range 1-51, 1 is best quality, default is 23
    # create_animation_ffmpeg(states, quality=10, output_path="media", color=True)  # quality range 1-51, 1 is best quality, default is 23
    print(f"\n\nTime taken: {time.time() - tic:.2f} seconds\n\n")
