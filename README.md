# goalNCA

Codebase for training Neural Cellular Automata (NCAs) to grow, morph, rotate, and translate target patterns conditioned on task vectors.


<p align="center">
  <img src="diagram.png" width="600">
</p>


<p align="center">
  <img src="animation.gif" width="600">
</p>

**Main directories:**
- `NCAs/`: NCA model implementations (MLP-based and LSTM-based variants).
- `src/trainers/`: Training loop and multiplexing experiment orchestration.
- `src/datasets/`: Dataset classes for pattern loading and transformation tasks.
- `src/visualisation/`: Visualization and animation utilities.
- `src/utils/`: General utility functions and plotting helpers.
- `src/experiments/`: Multiplexing experiment scripts.
- `datasets/`: Raw dataset files (emoji/pattern images).


---

## Installation

Install dependencies using [uv](https://docs.astral.sh/uv/) and the provided `pyproject.toml` or `uv.lock`.

First [install uv](https://docs.astral.sh/uv/getting-started/installation/), then simply type `uv sync`.

---

## Usage

### Training

To train an NCA, run:

```bash
uv run python train.py --conf config_growing.yml
```

**Supported tasks:**
- **Conditional Growth**: Grow patterns from seeds conditioned on task vectors.
- **Morphing**: Morph from one pattern to another.
- **Translation**: Translate patterns in space.

Available configuration files:
- `config_growing.yml`: Conditional growth of patterns from seeds.
- `config_morphing.yml`: Morphing between patterns.
- `config_translation.yml`: Translating patterns in space.

Results are saved to `results/{task_name}/{run_name}/`.

### Evaluation

To evaluate a trained model:

```bash
uv run python evaluate.py --path results/{task_name}/{run_name}/
```

---

### Configuration

Configuration files (`config_*.yml`) control all aspects of training, including:
- Task type and target patterns (emoji unicode, image paths, or dataset folders)
- Model architecture (layers, dimensions, activations, convolution mode)
- NCA dynamics (steps, noise, stochastic updates, boundary conditions)
- WandB experiment tracking
