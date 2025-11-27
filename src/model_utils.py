from copy import deepcopy
from functools import partial

import hydra
import torch
import torchvision
from utils.utils import prepare_noise_hook


def prepare_model(model_arch, device, num_models=1, compile_model: bool=False, gradient_noise: float=0, y_mean=None, y_std=None, atom_refs=None):
    # model_initializer = partial(model_arch.initializer, y_mean=y_mean, y_std=y_std, atom_refs=atom_refs)

    models = [hydra.utils.instantiate(model_arch.initializer, y_mean=y_mean, y_std=y_std, atom_refs=atom_refs) for _ in range(num_models)]
    models = [model.to(device) for model in models]
    if compile_model:
        models = [torch.compile(model) for model in models]

    return models
