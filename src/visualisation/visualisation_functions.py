import torch
import numpy as np
import wandb
from matplotlib import animation, pyplot as plt
import pathlib
import matplotlib.pyplot as plt
import time

from NCAs.utils import return_activation_function, compute_alive_mask


def is_one_hot(tensor):
    return bool(torch.all((tensor == 0) | (tensor == 1)))


def run_live_animation(
    config,
    model,
    initial_state,
    task_vector,
    device,
    emoji_name,
    activation_state_norm,
    headless=False,
    output_dir=None,
):
    """Run live animation showing NCA evolution. Pauses after config steps, continues on C/SPACE.

    task_vector can be a single tensor or a list of tensors (sequence mode).
    In sequence mode, each vector is applied for configured_steps before moving to the next.
    """
    import matplotlib.pyplot as plt
    import time
    embedding_dim = config["embedding_dim"]
    configured_steps = config["nca_steps_eval"] if "nca_steps_eval" in config else config["nca_steps"]

    # Normalize task_vector to a list for uniform handling
    is_sequence = isinstance(task_vector, list)
    if is_sequence:
        task_vector_list = task_vector
    else:
        task_vector_list = [task_vector]

    print("\n" + "=" * 60)
    print("LIVE NCA ANIMATION")
    print("=" * 60)
    if is_sequence:
        print(f"Sequence mode: {len(task_vector_list)} steps, {configured_steps} NCA steps each")
    elif "nca_steps_is_range" in config and config["nca_steps_is_range"]:
        print(f"Will run {configured_steps} steps (middle of range {config['nca_steps']}), then pause")
    else:
        print(f"Will run {configured_steps} steps, then pause")
    print("Press C or SPACE to continue after pause, Q to quit")
    if emoji_name:
        print(f"Pattern: {emoji_name}")
    print("=" * 60)

    _, _, H, W = initial_state.shape

    def _make_external_input(tv):
        tv_batch = tv.unsqueeze(0).to(device)
        return tv_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

    external_input = _make_external_input(task_vector_list[0])

    # Handle state_norm properly - use provided activation or create default
    if activation_state_norm is None:
        if config["state_norm"]:
            activation_state_norm = lambda x: torch.clamp(x, 0, 1)
        else:
            activation_state_norm = lambda x: x

    # Initialize state
    current_state = initial_state.clone()
    step_count = 0

    # Set up the plot
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create initial image
    initial_img = current_state[0, :embedding_dim].clip(0, 1).cpu().numpy().transpose(1, 2, 0)
    im = ax.imshow(initial_img)
    ax.set_title(f"Live NCA Evolution - Step {step_count}")
    ax.axis("off")

    plt.tight_layout()
    plt.show()

    def _run_steps(n_steps, ext_input, label=""):
        """Run n_steps of NCA with the given external input, updating the plot."""
        nonlocal current_state, step_count
        if label:
            print(f"\n>> {label}")
        for i in range(n_steps):
            with torch.no_grad():
                if config["alive_mask"]:
                    pre_life_mask = compute_alive_mask(current_state, config["alive_threshold"])

                if config["additive_update"]:
                    current_state = current_state + model(current_state, x_ext=ext_input, update_noise=0.0)
                else:
                    current_state = model(current_state, x_ext=ext_input, update_noise=0.0)

                if config["alive_mask"]:
                    post_life_mask = compute_alive_mask(current_state, config["alive_threshold"])
                    life_mask = (pre_life_mask & post_life_mask).float()
                    current_state = current_state * life_mask

                current_state = activation_state_norm(current_state)
                step_count += 1

                img = current_state[0, :embedding_dim].clip(0, 1).cpu().numpy().transpose(1, 2, 0)
                im.set_data(img)
                title = f"Step {step_count:,}"
                if label:
                    title += f" | {label}"
                ax.set_title(title)
                plt.draw()
                plt.pause(0.05)
                print(f"\r  Step: {step_count:,}", end="", flush=True)
        print()

    def _wait_for_key():
        """Wait for C/SPACE (continue) or Q (quit). Returns True to continue, False to quit."""
        print(f"\nPaused at step {step_count}. Press C/SPACE to continue, Q to quit")
        result = [None]

        def on_key(event):
            if event.key in (" ", "c", "C"):
                result[0] = True
            elif event.key in ("q", "Q"):
                result[0] = False

        cid = fig.canvas.mpl_connect("key_press_event", on_key)
        while result[0] is None:
            plt.pause(0.1)
        fig.canvas.mpl_disconnect(cid)
        return result[0]

    try:
        if is_sequence:
            # Sequence mode: play each vector for configured_steps, no pausing between
            for seq_idx, tv in enumerate(task_vector_list):
                ext = _make_external_input(tv)
                _run_steps(configured_steps, ext, label=f"Seq {seq_idx+1}/{len(task_vector_list)}")

            if not headless:
                # After sequence completes, allow replaying
                while True:
                    if not _wait_for_key():
                        break
                    for seq_idx, tv in enumerate(task_vector_list):
                        ext = _make_external_input(tv)
                        _run_steps(configured_steps, ext, label=f"Seq {seq_idx+1}/{len(task_vector_list)}")
        else:
            # Single vector mode: run configured_steps, pause, repeat on C/SPACE
            _run_steps(configured_steps, external_input)
            if not headless:
                while True:
                    if not _wait_for_key():
                        break
                    _run_steps(configured_steps, external_input)

    except KeyboardInterrupt:
        print(f"\n\nAnimation stopped after {step_count:,} steps")
    except Exception as e:
        print(f"\n\nAnimation error: {e}")
    finally:
        if headless and output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            img = current_state[0, :embedding_dim].clip(0, 1).cpu().numpy().transpose(1, 2, 0)
            out_path = os.path.join(output_dir, "final_frame.png")
            plt.imsave(out_path, img)
            print(f"Saved final frame to {out_path}")
        plt.ioff()
        plt.close("all")
        print(f"Final step count: {step_count:,}")
