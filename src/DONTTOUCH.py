import numpy as np
import torch
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

from utils import seed_everything


#call by:
#python3 src/DONTTOUCH.py +test.checkpoint_dir=models/folderName" +test.modelcount=n

#
#
#setup and dataloader refactored from run.py
#followed by validation and testing methods refactored from trainer.py
#
#


#setup like in run.py

@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)



        # Dataloader setup
    dm = hydra.utils.instantiate(cfg.dataset.init)
    test_dataloader = dm.test_dataloader()


    #load model weights or throw error if not found (weights saved from trainer.py)
    #docs: https://notes.kodekloud.com/docs/PyTorch/Building-and-Training-Models/Saving-and-loading-models?utm_source=chatgpt.com
    
    project_root = Path(get_original_cwd())
    modelWeights_dir = project_root / cfg.test.checkpoint_dir  

    models = []
    for i in range(cfg.test.modelcount):  
        modelWeights_path = modelWeights_dir / f"model_{i}.pt"

        if not modelWeights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {modelWeights_path}")

        model = hydra.utils.instantiate(cfg.model.init).to(device)
        if cfg.compile_model:
            model = torch.compile(model)

        state_dict = torch.load(modelWeights_path, map_location=device)
        model.load_state_dict(state_dict)

        models.append(model)


    #evaluation as in trainer.py
    for model in models:
        model.eval()
    test_losses = []

    with torch.no_grad():
        for x, targets in test_dataloader:
            x, targets = x.to(device), targets.to(device)
            preds = [model(x) for model in models]
            avg_preds = torch.stack(preds).mean(0)  # remember to adapt to ensemble tbd
            loss = torch.nn.functional.mse_loss(avg_preds, targets)
            test_losses.append(loss.item())

    test_mse = float(np.mean(test_losses))
    metrics = {"test_MSE": test_mse}

    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
