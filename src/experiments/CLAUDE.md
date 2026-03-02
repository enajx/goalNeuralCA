# Experiments

## multiplexing.py

**Motivation**: Investigate the relationship between NCA capacity and the number of patterns it must learn. How does final loss scale as we increase the number of target patterns from 1 to N? This helps understand the limits of pattern multiplexing in a single NCA model.

**Method**: For each n from 1 to N, randomly sample n patterns from a pool and train an NCA. Repeat multiple times per n to average out variance from pattern selection and training stochasticity.

**Usage**:
```bash
uv run python src/experiments/multiplexing.py --target-patterns datasets/emojis/ --nb-patterns-max 20 --runs-per-n 5 --epochs 5000
```

**Args**:
- `--target-patterns`: "emojis" (built-in 65 emoji list) or folder path (required unless --resume or --analyze-only)
- `--nb-patterns-max`: maximum number of patterns to test
- `--nb-patterns-step`: step size for number of patterns (default 1; e.g., `--nb-patterns-step 3` gives 1,4,7,...)
- `--runs-per-n`: independent runs per n value (different random pattern subsets)
- `--epochs`: override epochs from base config
- `--gpus`: comma-separated GPU IDs (e.g., "0,2,3"), uses all available if not specified
- `--resume <experiment_id>`: resume a cancelled experiment, skipping already-completed runs
- `--analyze-only --experiment-id <id>`: skip training, regenerate plots/analysis

**Output** (in `results/multiplexing/{datetime_id}/`):
- `runs/`: individual training outputs
- `plots/`: best_loss_vs_n.png, best_loss_vs_n_log.png, loss_vs_epoch.png, loss_vs_epoch_log.png
- `best_loss_vs_n.csv`, `mean_loss_vs_epoch.csv`
- `completed_runs.yml`: tracks which run keys have finished (for resume support)
- `run_manifest.yml`: stores the random pattern assignments per run (deterministic resume)

**Resuming a cancelled experiment**:
```bash
ls results/multiplexing/  # find experiment ID
uv run python src/experiments/multiplexing.py --resume 20260213_143052
```
All parameters (nb-patterns-max, nb-patterns-step, runs-per-n, epochs, pattern assignments) are restored from the saved config. `--epochs` cannot be overridden on resume.

**Features**: auto GPU detection, round-robin distribution across GPUs, resume support (skips completed runs), fail-fast on errors.
