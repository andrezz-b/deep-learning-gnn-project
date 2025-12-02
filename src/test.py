import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from utils import seed_everything


@hydra.main(config_path="../configs", config_name="test", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    # Instantiate DataModule
    dm = hydra.utils.instantiate(cfg.dataset.init)
    
    # Instantiate Model
    model = hydra.utils.instantiate(cfg.model.init).to(device)

    # Load Model Weights
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model file not found at {cfg.model_path}")
    
    print(f"Loading model from {cfg.model_path}")
    state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Test Loop
    test_loader = dm.test_dataloader()
    test_losses = []
    
    print("Starting testing...")
    with torch.no_grad():
        for x, targets in test_loader:
            x, targets = x.to(device), targets.to(device)
            
            # Forward pass
            preds = model(x)
            
            # Calculate loss (MSE)
            test_loss = torch.nn.functional.mse_loss(preds, targets)
            test_losses.append(test_loss.item())

    avg_test_loss = np.mean(test_losses)
    print(f"Test MSE: {avg_test_loss}")
    print(f"FINAL_TEST_MSE: {avg_test_loss}")

if __name__ == "__main__":
    main()
