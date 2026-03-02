"""
Multiplexing Experiment Script

Studies how NCA loss scales with the number of patterns (n=1 to N),
running multiple training runs per n to average out stochasticity.
"""

import os
import sys
import yaml
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from glob import glob
import shutil
import time
import random
import signal

# Global for cleanup on interrupt
_active_run_processes = {}

EMOJI_LIST = [
    "😊",
    "😂",
    "🔥",
    "👍",
    "😍",
    "🎉",
    "🌟",
    "💯",
    "🚀",
    "🎵",
    "🌈",
    "⭐",
    "🌺",
    "🎨",
    "🦄",
    "🍀",
    "🌙",
    "🌻",
    "🌊",
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐒",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🍎",
    "🍌",
    "🍓",
    "🍇",
    "🍉",
    "🍑",
    "🍒",
    "🥭",
    "🍍",
    "🥥",
    "🥝",
    "🍅",
    "🥑",
    "🥒",
    "🥕",
    "🌽",
    "🥔",
    "🍠",
    "🥐",
    "🌴",
    "😈",
    "🖤",
    "🤘",
    "🍬",
    "🦋",
    "🪲",
]


def cleanup_on_interrupt(signum, frame):
    """Terminate all running processes on Ctrl+C."""
    print("\n\nInterrupted. Terminating running processes...")
    for run_key, (proc, cfg, lf, lp) in _active_run_processes.items():
        if proc.poll() is None:
            proc.terminate()
        lf.close()
        if cfg.exists():
            cfg.unlink()
    sys.exit(1)


def get_available_gpus():
    """Return list of available GPU IDs, or empty list if no GPUs."""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    return []


def load_patterns_from_folder(folder_path):
    """Load pattern paths from a folder containing PNG files."""
    patterns = sorted(glob(os.path.join(folder_path, "*.png")))
    if not patterns:
        raise ValueError(f"No PNG files found in {folder_path}")
    return patterns


def create_run_config(base_config, patterns, seed, runs_dir, run_name, epochs, skip_animations):
    """Create a modified config for a specific run."""
    config = base_config.copy()
    config["target_patterns"] = patterns
    config["seed"] = seed
    config["saving_path"] = str(runs_dir)
    config["run_name"] = run_name
    config["wandb_mode"] = "disabled"
    config["skip_animations"] = skip_animations
    if epochs is not None:
        config["num_epochs"] = epochs
    return config


def find_completed_runs(runs_dir):
    """Find runs that have completed (have training_losses*.npy file).

    Note: train.py creates directories as {runs_dir}/{task}/{run_name}/,
    so we search two levels deep.
    """
    completed = set()
    if not runs_dir.exists():
        return completed
    # Search for loss files in {runs_dir}/*/{run_name}/ structure
    for loss_file in runs_dir.glob("*/*/training_losses_*.npy"):
        run_dir = loss_file.parent
        completed.add(run_dir.name)
    return completed


def record_completed_run(experiment_dir, run_key):
    """Record a completed run in completed_runs.yml."""
    path = experiment_dir / "completed_runs.yml"
    if path.exists():
        with open(path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader) or {}
    else:
        data = {}
    completed = set(data.get("completed", []))
    completed.add(run_key)
    data["completed"] = sorted(completed)
    with open(path, "w") as f:
        yaml.dump(data, f)


def check_for_failures(active_processes):
    """Check if any process has failed. Returns (failed_device, failed_process) or (None, None)."""
    for device_id, processes in active_processes.items():
        for p in processes:
            ret = p.poll()
            if ret is not None and ret != 0:
                return device_id, p
    return None, None


def cleanup_completed(active_processes):
    """Remove completed processes from tracking."""
    for device_id in list(active_processes.keys()):
        active_processes[device_id] = [p for p in active_processes[device_id] if p.poll() is None]


def wait_for_process_slot(active_processes, max_per_device, next_device_idx, num_devices):
    """Wait until there's an available slot, using round-robin distribution."""
    devices = list(active_processes.keys())
    while True:
        # Check for failures first
        failed_device, failed_proc = check_for_failures(active_processes)
        if failed_proc is not None:
            return None, next_device_idx  # Signal failure

        cleanup_completed(active_processes)

        # Round-robin: try starting from next_device_idx
        for i in range(num_devices):
            device_id = devices[(next_device_idx + i) % num_devices]
            if len(active_processes[device_id]) < max_per_device:
                return device_id, (next_device_idx + 1) % num_devices
        time.sleep(0.5)


def run_training(config_path, device_id, use_gpu, log_path):
    """Launch a training subprocess with output redirected to a log file."""
    env = os.environ.copy()
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [sys.executable, "train.py", "--conf", str(config_path)]
    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, cwd=Path(__file__).parent.parent.parent)
    return process, log_file


def collect_losses(runs_dir, n_values, runs_per_n):
    """Collect all loss data from completed runs.

    Note: train.py creates directories as {runs_dir}/{task}/{run_name}/,
    so we search with pattern */{n_str}_run{idx}_*.
    """
    num_n = len(n_values)
    best_losses = np.full((num_n, runs_per_n), np.nan)
    max_epochs = 0

    # First pass: find max_epochs and best losses
    for i, n in enumerate(n_values):
        n_str = f"n{n:02d}"
        for run_idx in range(runs_per_n):
            pattern = f"*/{n_str}_run{run_idx}_*"
            matching_dirs = list(runs_dir.glob(pattern))
            if matching_dirs:
                run_dir = matching_dirs[0]
                loss_files = list(run_dir.glob("training_losses_*.npy"))
                if loss_files:
                    losses = np.load(loss_files[0])
                    best_losses[i, run_idx] = np.min(losses)
                    max_epochs = max(max_epochs, len(losses))

    # Second pass: collect full loss curves
    all_losses = np.full((num_n, runs_per_n, max_epochs), np.nan)
    for i, n in enumerate(n_values):
        n_str = f"n{n:02d}"
        for run_idx in range(runs_per_n):
            pattern = f"*/{n_str}_run{run_idx}_*"
            matching_dirs = list(runs_dir.glob(pattern))
            if matching_dirs:
                run_dir = matching_dirs[0]
                loss_files = list(run_dir.glob("training_losses_*.npy"))
                if loss_files:
                    losses = np.load(loss_files[0])
                    all_losses[i, run_idx, : len(losses)] = losses

    return best_losses, all_losses


def create_plots(best_losses, all_losses, n_values, experiment_dir):
    """Create and save analysis plots (linear and log scale)."""
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mean_best = np.nanmean(best_losses, axis=1)
    std_best = np.nanstd(best_losses, axis=1)

    # Plot 1 & 2: Best Loss vs Number of Patterns (linear and log)
    for scale, suffix in [("linear", ""), ("log", "_log")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(n_values, mean_best, yerr=std_best, fmt="o-", capsize=5)
        ax.fill_between(n_values, mean_best - std_best, mean_best + std_best, alpha=0.3)
        ax.set_xlabel("Number of Patterns (n)")
        ax.set_ylabel("Best Loss")
        ax.set_title(f"Best Loss vs Number of Patterns ({scale})")
        ax.set_yscale(scale)
        ax.grid(True, alpha=0.3)
        fig.savefig(plots_dir / f"best_loss_vs_n{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Plot 3 & 4: Loss vs Epoch (linear and log)
    for scale, suffix in [("linear", ""), ("log", "_log")]:
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.cm.viridis
        for n in range(all_losses.shape[0]):
            mean_curve = np.nanmean(all_losses[n], axis=0)
            std_curve = np.nanstd(all_losses[n], axis=0)
            valid_mask = ~np.isnan(mean_curve)
            epochs = np.arange(len(mean_curve))
            color = cmap(n / max(1, all_losses.shape[0] - 1))
            ax.plot(epochs[valid_mask], mean_curve[valid_mask], color=color, label=f"n={n_values[n]}")
            if scale == "linear":
                ax.fill_between(
                    epochs[valid_mask],
                    (mean_curve - std_curve)[valid_mask],
                    (mean_curve + std_curve)[valid_mask],
                    color=color,
                    alpha=0.1,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss vs Epoch by Number of Patterns ({scale})")
        ax.set_yscale(scale)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(plots_dir / f"loss_vs_epoch{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return plots_dir


def save_aggregated_results(best_losses, all_losses, n_values, experiment_dir):
    """Save aggregated results as CSV files."""
    num_n, runs_per_n = best_losses.shape

    # best_loss_vs_n.csv: columns = run0, run1, ..., mean, std; rows = n values
    header = [f"run{i}" for i in range(runs_per_n)] + ["mean", "std"]
    rows = []
    for i in range(num_n):
        row = list(best_losses[i]) + [np.nanmean(best_losses[i]), np.nanstd(best_losses[i])]
        rows.append(row)
    with open(experiment_dir / "best_loss_vs_n.csv", "w") as f:
        f.write("n," + ",".join(header) + "\n")
        for i, row in enumerate(rows):
            f.write(f"{n_values[i]}," + ",".join(f"{v:.6f}" for v in row) + "\n")

    # mean_loss_vs_epoch.csv: columns = epoch; rows = n values
    mean_curves = np.nanmean(all_losses, axis=1)
    with open(experiment_dir / "mean_loss_vs_epoch.csv", "w") as f:
        f.write("n," + ",".join(f"epoch{i}" for i in range(mean_curves.shape[1])) + "\n")
        for i in range(num_n):
            f.write(f"{n_values[i]}," + ",".join(f"{v:.6f}" if not np.isnan(v) else "" for v in mean_curves[i]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multiplexing experiment for NCA")
    parser.add_argument("--config", type=str, default="config_growing.yml", help="Base config file")
    parser.add_argument(
        "--target-patterns",
        type=str,
        default=None,
        help="Folder path OR 'emojis' to use built-in emoji list (required unless --resume or --analyze-only)",
    )
    parser.add_argument("--nb-patterns-max", type=int, default=20, help="Maximum number of patterns to test")
    parser.add_argument(
        "--nb-patterns-step", type=int, default=1, help="Step size for number of patterns (e.g., 3 gives 1,4,7,...)"
    )
    parser.add_argument("--runs-per-n", type=int, default=5, help="Number of runs per n value")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs (uses config value if not specified)")
    parser.add_argument("--analyze-only", action="store_true", help="Skip training, only run analysis")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID for analyze-only mode")
    parser.add_argument("--resume", type=str, default=None, help="Experiment ID to resume (e.g., '20260213_143052')")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,2,3'). Uses all available if not specified.",
    )
    parser.add_argument(
        "--animations",
        action="store_true",
        default=False,
        help="Generate .mp4 animations for each run (default: skip animations, only produce static PNGs)",
    )
    args = parser.parse_args()

    # Determine pattern pool (not needed for --resume or --analyze-only)
    if not args.resume and not args.analyze_only:
        if args.target_patterns is None:
            parser.error("--target-patterns is required unless using --resume or --analyze-only")
        if args.target_patterns == "emojis":
            pattern_pool = EMOJI_LIST.copy()
        else:
            pattern_pool = load_patterns_from_folder(args.target_patterns)

        if len(pattern_pool) < args.nb_patterns_max:
            print(f"Warning: pattern pool ({len(pattern_pool)}) < nb_patterns_max ({args.nb_patterns_max})")
            args.nb_patterns_max = len(pattern_pool)

    # Setup experiment directory
    if args.analyze_only:
        if args.experiment_id is None:
            raise ValueError("--experiment-id required for --analyze-only mode")
        experiment_dir = Path("results/multiplexing") / args.experiment_id
        if not experiment_dir.exists():
            raise ValueError(f"Experiment directory not found: {experiment_dir}")
    elif args.resume:
        experiment_dir = Path("results/multiplexing") / args.resume
        if not experiment_dir.exists():
            raise ValueError(f"Experiment directory not found: {experiment_dir}")
        if args.epochs is not None:
            parser.error("--epochs cannot be used with --resume (epochs are restored from the original experiment)")
        experiment_id = args.resume
    else:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("results/multiplexing") / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    with open(args.config) as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    if not args.analyze_only:
        if args.resume:
            # Load existing experiment config and manifest
            with open(experiment_dir / "experiment_config.yml") as f:
                experiment_config = yaml.load(f, Loader=yaml.FullLoader)
            args.nb_patterns_max = experiment_config.get("nb_patterns_max", experiment_config.get("n_max"))
            args.nb_patterns_step = experiment_config.get("nb_patterns_step", experiment_config.get("n_step", 1))
            args.runs_per_n = experiment_config["runs_per_n"]
            args.epochs = experiment_config["epochs"]
            if "animations" in experiment_config:
                args.animations = experiment_config["animations"]

            with open(experiment_dir / "run_manifest.yml") as f:
                run_manifest = yaml.load(f, Loader=yaml.FullLoader)

            print(f"Resuming experiment {args.resume} (epochs={args.epochs})")
        else:
            # Save experiment config — resolve epochs now so resume is reliable
            experiment_config = {
                "base_config": args.config,
                "target_patterns": args.target_patterns,
                "nb_patterns_max": args.nb_patterns_max,
                "nb_patterns_step": args.nb_patterns_step,
                "runs_per_n": args.runs_per_n,
                "epochs": args.epochs if args.epochs is not None else base_config["num_epochs"],
                "animations": args.animations,
                "pattern_pool_size": len(pattern_pool),
                "experiment_id": experiment_id,
            }
            with open(experiment_dir / "experiment_config.yml", "w") as f:
                yaml.dump(experiment_config, f)

            # Generate run manifest
            run_manifest = {}
            master_seed = random.randint(0, 2**31)
            rng = random.Random(master_seed)

            for n in [1] + [n for n in range(args.nb_patterns_step, args.nb_patterns_max + 1, args.nb_patterns_step) if n != 1]:
                for run_idx in range(args.runs_per_n):
                    run_key = f"n{n:02d}_run{run_idx}"
                    sampled_patterns = rng.sample(pattern_pool, n)
                    run_seed = rng.randint(0, 2**31)
                    run_manifest[run_key] = {
                        "n": n,
                        "run_idx": run_idx,
                        "patterns": sampled_patterns,
                        "seed": run_seed,
                    }

            with open(experiment_dir / "run_manifest.yml", "w") as f:
                yaml.dump(run_manifest, f)

        # Load completed runs from tracking file + filesystem fallback
        completed_runs = set()
        completed_file = experiment_dir / "completed_runs.yml"
        if completed_file.exists():
            with open(completed_file) as f:
                data = yaml.load(f, Loader=yaml.FullLoader) or {}
            completed_runs = set(data.get("completed", []))
        completed_dirs = find_completed_runs(runs_dir)
        completed_runs = completed_runs | completed_dirs

        n_values = [1] + [n for n in range(args.nb_patterns_step, args.nb_patterns_max + 1, args.nb_patterns_step) if n != 1]
        total_runs = len(n_values) * args.runs_per_n
        print(f"Completed: {len(completed_runs)}/{total_runs} runs already done")

        # Determine device setup
        available_gpus = get_available_gpus()
        if args.gpus is not None:
            devices = [int(g) for g in args.gpus.split(",")]
            use_gpu = True
            print(f"Using specified GPUs: {devices}")
        elif len(available_gpus) > 0:
            devices = available_gpus
            use_gpu = True
            print(f"Using GPUs: {devices}")
        else:
            devices = [0]
            use_gpu = False
            print("No GPUs available, running on CPU")

        active_processes = {d: [] for d in devices}
        pending_runs = []

        # Queue all pending runs
        for run_key, run_info in run_manifest.items():
            # Check if any directory matching this run exists and is complete
            matching = [d for d in completed_runs if d.startswith(run_key)]
            if matching:
                continue
            pending_runs.append((run_key, run_info))

        total_to_run = len(pending_runs)
        print(f"Runs to execute: {total_to_run}")

        # Execute runs with round-robin GPU distribution
        global _active_run_processes
        signal.signal(signal.SIGINT, cleanup_on_interrupt)
        signal.signal(signal.SIGTERM, cleanup_on_interrupt)
        run_processes = _active_run_processes
        next_device_idx = 0
        num_devices = len(devices)
        started_count = 0

        for run_key, run_info in pending_runs:
            # Wait for available slot (round-robin)
            device_id, next_device_idx = wait_for_process_slot(active_processes, 1, next_device_idx, num_devices)

            # Check for failure signal
            if device_id is None:
                print("\nERROR: A running process failed. Aborting experiment.")
                for proc_key, (proc, cfg, lf, lp) in run_processes.items():
                    if proc.poll() is None:
                        proc.terminate()
                    lf.close()
                    if cfg.exists():
                        cfg.unlink()
                sys.exit(1)

            # Create config for this run
            run_name = f"{run_key}_{datetime.now().strftime('%H%M%S')}"

            config = create_run_config(
                base_config, run_info["patterns"], run_info["seed"], runs_dir, run_name, args.epochs, not args.animations
            )

            # Write temporary config
            temp_config_path = experiment_dir / f"temp_config_{run_key}.yml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)

            started_count += 1
            print(f"[{started_count}/{total_to_run}] Starting {run_key} on GPU {device_id}")
            log_path = experiment_dir / f"log_{run_key}.txt"
            process, log_file = run_training(temp_config_path, device_id, use_gpu, log_path)
            active_processes[device_id].append(process)
            run_processes[run_key] = (process, temp_config_path, log_file, log_path)

        # Wait for all processes to complete, checking for failures
        print(f"\nAll {total_to_run} runs launched. Waiting for completion...")
        completed_keys = set()
        while len(completed_keys) < len(run_processes):
            for run_key, (process, temp_config_path, log_file, log_path) in run_processes.items():
                if run_key in completed_keys:
                    continue
                ret = process.poll()
                if ret is not None:
                    completed_keys.add(run_key)
                    log_file.close()
                    if temp_config_path.exists():
                        temp_config_path.unlink()
                    if ret != 0:
                        error_tail = ""
                        if log_path.exists():
                            with open(log_path) as f:
                                lines = f.readlines()
                                error_tail = "".join(lines[-20:])
                        print(f"FAILED: {run_key}")
                        print(f"Log tail:\n{error_tail}")
                        print("\nAborting experiment due to failure. Terminating other runs...")
                        for k, (p, cfg, lf, lp) in run_processes.items():
                            if k not in completed_keys and p.poll() is None:
                                p.terminate()
                            lf.close()
                            if cfg.exists():
                                cfg.unlink()
                        sys.exit(1)
                    else:
                        record_completed_run(experiment_dir, run_key)
                        if log_path.exists():
                            log_path.unlink()
                        print(f"[{len(completed_keys)}/{total_to_run}] Completed: {run_key}")
            time.sleep(0.5)

        # Clear global after all complete
        _active_run_processes.clear()

    # Analysis phase
    print("\n=== Analysis Phase ===")

    # Load experiment config for analyze-only mode
    if args.analyze_only:
        with open(experiment_dir / "experiment_config.yml") as f:
            experiment_config = yaml.load(f, Loader=yaml.FullLoader)
        args.nb_patterns_max = experiment_config.get("nb_patterns_max", experiment_config.get("n_max"))
        args.nb_patterns_step = experiment_config.get("nb_patterns_step", experiment_config.get("n_step", 1))
        args.runs_per_n = experiment_config["runs_per_n"]

    n_values = [1] + [n for n in range(args.nb_patterns_step, args.nb_patterns_max + 1, args.nb_patterns_step) if n != 1]
    best_losses, all_losses = collect_losses(runs_dir, n_values, args.runs_per_n)

    # Save aggregated results
    save_aggregated_results(best_losses, all_losses, n_values, experiment_dir)
    print(f"Saved aggregated .npy files to {experiment_dir}")

    # Create plots
    plots_dir = create_plots(best_losses, all_losses, n_values, experiment_dir)
    print(f"Saved plots to {plots_dir}")

    # Print summary
    print("\n=== Summary ===")
    for i, n in enumerate(n_values):
        mean_loss = np.nanmean(best_losses[i])
        std_loss = np.nanstd(best_losses[i])
        print(f"n={n:2d}: best_loss = {mean_loss:.6f} +/- {std_loss:.6f}")

    print(f"\nExperiment completed. Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
