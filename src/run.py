import os
from itertools import chain

import hydra
import torch
from omegaconf import OmegaConf

from utils import seed_everything


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

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    model = hydra.utils.instantiate(cfg.model.init).to(device)

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(
        cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device
    )

    results = trainer.train(**cfg.trainer.train)
    results = torch.Tensor(results)

    if cfg.save_model:
        save_dir = cfg.model_save_dir
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(hydra.utils.get_original_cwd(), save_dir)
        os.makedirs(save_dir, exist_ok=True)

        if len(models) == 1:
            save_path = os.path.join(save_dir, "model.pt")
            torch.save(models[0].state_dict(), save_path)
            if hasattr(logger, "save_artifact"):
                logger.save_artifact(save_path)
        else:
            for i, m in enumerate(models):
                save_path = os.path.join(save_dir, f"model_{i}.pt")
                torch.save(m.state_dict(), save_path)
                if hasattr(logger, "save_artifact"):
                    logger.save_artifact(save_path)


if __name__ == "__main__":
    main()
