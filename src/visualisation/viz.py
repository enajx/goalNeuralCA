import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_states(states_to_viz, frames=None, interval=200, filename=None, cmap=None):
    if frames is None:
        frames = range(states_to_viz.shape[0])
    elif isinstance(frames, int):
        frames = range(min(frames, states_to_viz.shape[0]))

    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(states_to_viz[0].transpose(1, 2, 0), cmap=cmap)
    ax.axis("off")

    # Adjust the subplot to remove white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update(frame):
        img.set_array(states_to_viz[frame].transpose(1, 2, 0))
        ax.set_title(f"Step {frame}", fontsize=10)
        return [img]

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    if filename is None:
        plt.show()
    else:
        ani.save(filename, writer="ffmpeg", dpi=300)


def animate_all_hidden_states(states, frames=None, interval=200, filename=None, cmap=None):
    """
    Animate all hidden states over time and save as MP4

    Args:
        states: tensor of shape (timesteps, channels, height, width)
        frames: which frames to include in animation
        interval: time interval between frames in ms
        filename: if provided, save animation to this file
        cmap: colormap for visualization
    """

    timesteps, channels, height, width = states.shape

    if frames is None:
        frames = range(timesteps)
    elif isinstance(frames, int):
        frames = range(min(frames, timesteps))

    # Set up the figure with subplots for each channel
    cols = min(4, channels - 1)  # Ensure at most 4 columns
    rows = 1 if channels - 1 <= 4 else -(-(channels - 1) // cols)  # Single row if <= 4

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axs = np.array([axs])  # Make it indexable
    elif rows == 1:
        axs = np.atleast_1d(axs)  # Ensure it's iterable

    # Adjust the subplot to remove white space
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.tight_layout()

    # Initialize with first frame
    images = []
    for i in range(1, channels):
        row_idx = (i - 1) // cols if rows > 1 else 0
        col_idx = (i - 1) % cols
        ax = axs[row_idx, col_idx] if rows > 1 else axs[i - 1]
        img = ax.imshow(states[0, i], animated=True, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Channel {i}")
        ax.axis("off")
        images.append(img)

    # Hide unused subplots
    if rows > 1:
        for i in range(channels, rows * cols + 1):
            row_idx = (i - 1) // cols
            col_idx = (i - 1) % cols
            if row_idx < rows and col_idx < cols:
                axs[row_idx, col_idx].axis("off")

    def update(frame_idx):
        frame = frames[frame_idx]
        for i, img in enumerate(images):
            img.set_array(states[frame, i + 1])
        fig.suptitle(f"Hidden States Evolution - Step {frame}", fontsize=14)
        return images

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    if filename is not None:
        ani.save(filename, writer="ffmpeg", dpi=150)
        print(f"Animation saved to {filename}")
    else:
        plt.show()

    return ani


def animate_hidden_channels(hidden_states, frames=None, interval=200, filename=None):
    """
    Animate hidden channels in a grid layout.

    Args:
        hidden_states: [time_steps, num_channels, H, W] tensor
        frames: frames to animate (default: all)
        interval: milliseconds between frames
        filename: output filename for video
    """
    if frames is None:
        frames = range(hidden_states.shape[0])
    elif isinstance(frames, int):
        frames = range(min(frames, hidden_states.shape[0]))

    num_channels = hidden_states.shape[1]
    cols = min(4, num_channels)
    rows = (num_channels + cols - 1) // cols

    # Ensure figure dimensions work with ffmpeg (need even pixel dimensions)
    fig_width = cols * 2.5
    fig_height = rows * 2.5
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if rows == 1:
        axs = np.atleast_1d(axs)

    # Initialize images for each subplot
    images = []
    for i in range(num_channels):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        img = ax.imshow(
            hidden_states[0, i],
            cmap="viridis",
            vmin=hidden_states[:, i].min(),
            vmax=hidden_states[:, i].max(),
        )
        ax.axis("off")
        ax.set_title(f"Hidden Ch {i+1}")
        images.append(img)

    # Hide unused subplots
    for i in range(num_channels, rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.axis("off")

    plt.tight_layout()

    def update(frame):
        for i in range(num_channels):
            images[i].set_array(hidden_states[frame, i])
        fig.suptitle(f"Hidden Channels - Step {frame}", fontsize=12)
        return images

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    if filename is None:
        plt.show()
    else:
        try:
            # Adjust figure height slightly to ensure even pixel dimensions
            dpi = 150
            pixel_height = int(fig_height * dpi)
            if pixel_height % 2 == 1:
                # Adjust figure height to make pixel height even
                fig_height_adjusted = (pixel_height + 1) / dpi
                fig.set_size_inches(fig_width, fig_height_adjusted)

            ani.save(filename, writer="ffmpeg", dpi=dpi)
            plt.close(fig)
        except Exception as e:
            print(f"\nWarning: Failed to save animation with ffmpeg: {e}")
            print(f"Skipping animation save for {filename}\n")
            plt.close(fig)

    return ani
