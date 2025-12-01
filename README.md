# GNN Introduction

This project provides an introduction to Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric on the dataset QM9.

## Installation

To run this project, you need to install the required Python packages. You can install them using pip:

```bash
# It is recommended to install PyTorch first, following the official instructions
# for your specific hardware (CPU or GPU with a specific CUDA version).
# See: https://pytorch.org/get-started/locally/

# For example, for a recent CUDA version:
# pip install torch torchvision torchaudio

# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# After installing PyTorch, install PyTorch Geometric.
# The exact command depends on your PyTorch and CUDA versions.
# Please refer to the PyTorch Geometric installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.7 and CUDA 11.8
# pip install torch_geometric

# Then, install the other required packages:
pip install hydra-core omegaconf wandb pytorch-lightning numpy tqdm
```

## How to Run

The main entry point for this project is `src/run.py`. It uses `hydra` for configuration management. Hydra is a broadly used and highly respected so I recommend using it. You can find a guide to it here https://medium.com/@jennytan5522/introduction-to-hydra-configuration-for-python-646e1dd4d1e9.

To run the code, execute the following command from the root of the project:

```bash
python src/run.py
```

You can override the default configuration by passing arguments from the command line. For example, to use a different model configuration:

```bash
python src/run.py model=gcn
```

The configuration files are located in the `configs/` directory.

## Experiments

Experiments are reproducible runs with specific configurations. They are stored in `configs/experiments/`.

### Creating an Experiment

To create a new experiment, add a `.yaml` file to `configs/experiments/`.
For example, create `configs/experiments/my_experiment.yaml`:

```yaml
# @package _global_

defaults:
  - override /model: gin
  - override /optimizer: adam
  - override /scheduler: cosine

model:
  init:
    hidden_channels: 128
    dropout: 0.5

trainer:
  train:
    total_epochs: 100

save_model: true
```

The `# @package _global_` directive allows the experiment config to override parameters at the global level.

### Running an Experiment

To run an experiment, use the `+experiments` command line argument:

```bash
python src/run.py +experiments=my_experiment
```

## Model Saving

When `save_model: true` is set in your configuration (either in `configs/run.yaml` or an experiment file), trained model files will be saved to the project-level directory `saved_models/` by default. The path is controlled by the `model_save_dir` config entry in `configs/run.yaml`.

If you are using the default `WandBLogger` (Weights & Biases), the code also uploads the saved model files as W&B artifacts for the active run. This happens automatically when `save_model: true` and W&B logging is enabled in your config.

Examples:

- Enable model saving globally (in `configs/run.yaml`):

```yaml
save_model: true
model_save_dir: saved_models
```

- Or enable it per experiment (e.g. `configs/experiments/gcn_simple.yaml`):

```yaml
save_model: true
```

Where the files are stored locally (relative to the project root):

```
./saved_models/model.pt
```

And the same files will be attached to the WandB run as artifacts.

## Improving the predictive accuracy
There are many ways to improve the GNN. Please try to get the validation error (MSE) as low as possible. I have not implemented the code to run on the test data. That is for you to do, but please wait until you have the final model.
Here are some great resources:
- Try different GNN architectures and layers see (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)
- Try different optimizers and schedulers
- Tune hyperparameters (especially learning rate, layers, and hidden units)
- Use advanced regularization techniques such as https://openreview.net/forum?id=xkljKdGe4E#discussion
- You can try changing the generated features of the dataloader

