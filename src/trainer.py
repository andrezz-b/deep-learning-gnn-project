from typing import Callable, cast

import numpy as np
import torch
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from logger import WandBLogger
from models import GIN
from qm9 import QM9DataModule


class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion: torch.nn.MSELoss,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        models: list[torch.nn.Module],
        logger: WandBLogger,
        datamodule: QM9DataModule,
    ):
        self.device: torch.device = device
        self.models: list[Module] = models

        # Optim related things
        self.supervised_criterion: MSELoss = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer: Optimizer = optimizer(params=all_params)
        self.scheduler: LRScheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        # self.logger.log_dict()
        results = []
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses: list[torch.Tensor] = [
                    self.supervised_criterion(model(x), targets)
                    for model in self.models
                ]
                supervised_loss = torch.stack(supervised_losses).sum()
                supervised_losses_logged.append(
                    supervised_loss.detach().item() / len(self.models)
                )
                loss = supervised_loss
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            # collect per-epoch supervised loss for return
            results.append(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)
        return results


class GraphMixup:
    def __init__(
        self,
        supervised_criterion: torch.nn.MSELoss,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        models: list[torch.nn.Module],
        logger: WandBLogger,
        datamodule: QM9DataModule,
        unsupervised_weight: float = 1.0,  # Max weight for unsupervised loss
        alpha: float = 1.0,
        k_perturbations: int = 10
    ):
        assert isinstance(models[0], GIN), "Only GIN is supported"
        self.device = device
        self.models = models
        self.supervised_criterion = supervised_criterion
        self.alpha = alpha
        self.k_perturbations = k_perturbations

        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.train_dataloader = datamodule.train_dataloader()
        # Use cycle to ensure we don't stop early if unlabeled data > labeled data
        self.unsupervised_train_dataloader = datamodule.unsupervised_train_dataloader()

        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.logger = logger
        self.unsupervised_weight = unsupervised_weight

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def get_consistency_weight(self, epoch: int, total_epochs: int) -> float:
        # First 50% of epochs: weight = 0. Last 50%: sigmoid-like ramp-up to self.unsupervised_weight.
        rampup_start = total_epochs // 2
        if epoch <= rampup_start:
            return 0.0

        rampup_length = total_epochs - rampup_start
        if rampup_length <= 0:
            return self.unsupervised_weight

        p = float(epoch - rampup_start) / float(rampup_length)  # in (0, 1]
        p = min(max(p, 0.0), 1.0)
        return self.unsupervised_weight * np.exp(-5.0 * (1.0 - p) * (1.0 - p))

    def train(self, total_epochs: int, validation_interval: int):
        results = []

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            epoch_loss_log = []

            # w(t): Ramp-up weight for unsupervised loss
            w_t = self.get_consistency_weight(epoch, total_epochs)

            for (x_label, targets_label), (x_unlabel, targets_unlabel) in zip(
                self.train_dataloader, self.unsupervised_train_dataloader
            ):
                x_label, targets_label = x_label.to(self.device), targets_label.to(self.device)
                x_unlabel, targets_unlabel = x_unlabel.to(self.device), targets_unlabel.to(self.device)

                self.optimizer.zero_grad()
                model = cast(GIN, self.models[0])

                rand_index = np.random.randint(0, 2)
                # --- 1. GNN Forward Pass (Supervised) ---
                # Standard training on labeled data
                if rand_index == 0:
                    pred_gnn = model(x_label)
                    total_loss = self.supervised_criterion(pred_gnn, targets_label)
                else:
                    # --- 2. Generate Pseudo-Labels (K-Perturbations) ---
                    # We perform K forward passes with dropout enabled (model.train())
                    # to create "noisy" predictions, then average them.
                    with torch.no_grad():
                        model.train() # Ensure dropout is active for perturbations
                        perturb_preds = []
                        for _ in range(self.k_perturbations):
                            perturb_preds.append(model(x_unlabel))
                        
                        # Average the K predictions to get stable pseudo-targets
                        pseudo_targets = torch.stack(perturb_preds).mean(dim=0)
                        
                        # Detach to ensure no gradients flow through pseudo-label generation
                        pseudo_targets = pseudo_targets.detach()

                    # --- 3. FCN Forward Pass (Supervised Mixup) ---
                    # Manifold Mixup on labeled data
                    # Model returns: pred, y_a, y_b, lam
                    pred_sup, y_a, y_b, lam = model(
                        x_label, mixup=True, target=targets_label, alpha=self.alpha
                    )

                    # Mixup Loss calculation: lam * Loss(pred, y_a) + (1-lam) * Loss(pred, y_b)
                    loss_fcn_sup = lam * self.supervised_criterion(pred_sup, y_a) + (
                        1 - lam
                    ) * self.supervised_criterion(pred_sup, y_b)

                    # --- 4. FCN Forward Pass (Unsupervised Mixup) ---
                    # Manifold Mixup on unlabeled data using Pseudo-Labels
                    pred_unsup, y_a_u, y_b_u, lam_u = model(
                        x_unlabel, mixup=True, target=pseudo_targets, alpha=self.alpha
                    )

                    loss_fcn_unsup = lam_u * self.supervised_criterion(
                        pred_unsup, y_a_u
                    ) + (1 - lam_u) * self.supervised_criterion(pred_unsup, y_b_u)

                    # --- 5. Total Loss Combination ---
                    # Loss = L_GNN + L_FCN_Supervised + w(t) * L_FCN_Unsupervised
                    total_loss = loss_fcn_sup + (w_t * loss_fcn_unsup)

                loss: torch.Tensor = total_loss
                loss.backward()
                self.optimizer.step()

                epoch_loss_log.append(loss.detach().item())

            self.scheduler.step()
            avg_loss = np.mean(epoch_loss_log)
            results.append(avg_loss)

            summary_dict = {"train_loss": avg_loss, "w_t": w_t}
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

        return results
