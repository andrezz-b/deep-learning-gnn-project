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

from logger import WandBLogger
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
        k_perturbations: int = 10,
        rampup_start_epoch: int = 500,  # Start ramping up w(t) here
        rampup_end_epoch: int = 1000,   # Reach max w(t) here
    ):
        assert isinstance(models[0], GIN), "Only GIN is supported"
        self.device = device
        self.models = models
        self.supervised_criterion = supervised_criterion
        self.alpha = alpha
        self.k_perturbations = k_perturbations
        
        # Ramp-up Configuration
        self.rampup_start_epoch = rampup_start_epoch
        self.rampup_end_epoch = rampup_end_epoch
        self.unsupervised_weight = unsupervised_weight

        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.train_dataloader = datamodule.train_dataloader()
        self.unsupervised_train_dataloader = datamodule.unsupervised_train_dataloader()

        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
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

    def get_consistency_weight(self, epoch: int) -> float:
        """
        Calculates w(t) based on the current epoch and the configured ramp-up schedule.
        Schedule: 0.0 -> ramp-up (sigmoid) -> max_weight
        """
        if epoch < self.rampup_start_epoch:
            return 0.0
        
        if epoch >= self.rampup_end_epoch:
            return self.unsupervised_weight

        # Normalize epoch to [0, 1] range relative to the ramp-up window
        rampup_length = self.rampup_end_epoch - self.rampup_start_epoch
        if rampup_length == 0: # Avoid division by zero
            return self.unsupervised_weight
            
        p = (epoch - self.rampup_start_epoch) / rampup_length
        p = min(max(p, 0.0), 1.0)
        
        # Sigmoid-like ramp-up function (from Mean Teacher / GraphMix paper)
        return self.unsupervised_weight * np.exp(-5.0 * (1.0 - p) * (1.0 - p))

    def train(self, total_epochs: int, validation_interval: int):
        # Assertion to ensure the ramp-up schedule makes sense for this run
        if self.rampup_end_epoch > total_epochs:
            print(f"Warning: Ramp-up end ({self.rampup_end_epoch}) is larger than total_epochs ({total_epochs}). "
                  f"Unsupervised loss will never reach full weight.")
        assert self.rampup_start_epoch < self.rampup_end_epoch, \
            f"Ramp-up start ({self.rampup_start_epoch}) must be before end ({self.rampup_end_epoch})"

        results = []

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            epoch_loss_log = []
            
            # Calculate w(t) for this epoch
            w_t = self.get_consistency_weight(epoch)

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

class MeanTeacherTrainer:
    def __init__(
            self,
            supervised_criterion,
            optimizer,
            scheduler,
            device,
            models,
            logger,
            datamodule,
            unsupervised_loss_weight: float = 0.0,
            early_stopping: dict | None = None,
    ):
        self.device = device
        self.models = models

        # Optimization related things
        self.supervised_criterion = supervised_criterion
        all_params = [
            p for m in self.models for p in m.parameters() if p.requires_grad
        ] # only student params get optimized
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.unsupervised_train_dataloader = None
        if hasattr(datamodule, "unsupervised_train_dataloader"):
            self.unsupervised_train_dataloader = (
                datamodule.unsupervised_train_dataloader()
            )

        # Logging
        self.logger = logger
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.early_stopping_cfg = early_stopping or {}
        self.early_stopping_monitor = self.early_stopping_cfg.get("monitor", "val_MSE")
        self.early_stopping_patience = self.early_stopping_cfg.get("patience", 0)
        self.early_stopping_min_delta = self.early_stopping_cfg.get("min_delta", 0.0)

    def _forward(self, model, batch, prefer_teacher: bool = False):
        if prefer_teacher:
            use_teacher = (
                    getattr(model, "teacher", None) is not None
            )  # checking if the model has an attribute called "teacher"
            if use_teacher:
                try:
                    return model(batch, use_teacher=True)
                except TypeError:
                    pass
        return model(batch)

    def _update_mean_teachers(self):
        for model in self.models:
            update_teacher = getattr(model, "update_teacher", None)
            if callable(update_teacher):
                update_teacher()

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # ensemble prediction, averaging across models
                preds = [
                    self._forward(model, x, prefer_teacher=True)
                    for model in self.models
                ]
                avg_preds = torch.stack(preds).mean(0)

                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        
        results = []
        best_metric = float("inf")
        epochs_since_improvement = 0

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            effective_unsup_weight = self.unsupervised_loss_weight
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            consistency_losses_logged = []
            unsupervised_iterator = None
            if (
                    self.unsupervised_train_dataloader is not None
                    and self.unsupervised_loss_weight > 0
            ):
                unsupervised_iterator = iter(self.unsupervised_train_dataloader)
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
            
                supervised_losses = [
                    self.supervised_criterion(self._forward(model, x), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(
                    supervised_loss.detach().item() / len(self.models)
                )  
                consistency_loss = None
                if unsupervised_iterator is not None:
                    try:
                        unsup_batch = next(unsupervised_iterator)
                    except StopIteration:
                        unsupervised_iterator = iter(self.unsupervised_train_dataloader)
                        unsup_batch = next(unsupervised_iterator)

                    unsup_data, _ = unsup_batch
                    unsup_data = unsup_data.to(self.device)
                    consistency_terms = []
                    for model in self.models:
                        if not hasattr(model, "teacher"):
                            continue
                        try:
                            teacher_preds = model(unsup_data, use_teacher=True)
                        except TypeError:
                            continue
                        student_preds = model(unsup_data)
                        consistency_terms.append(
                            torch.nn.functional.mse_loss(
                                student_preds, teacher_preds.detach()
                            )
                        )
                    if consistency_terms:
                        consistency_loss = sum(consistency_terms) / len(
                            consistency_terms
                        )
                        consistency_losses_logged.append(
                            consistency_loss.detach().item()
                        )

                loss = supervised_loss
                if consistency_loss is not None:
                    loss = loss + effective_unsup_weight * consistency_loss
                loss.backward()  
                self.optimizer.step()
                self._update_mean_teachers()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            if consistency_losses_logged:
                consistency_losses_logged = np.mean(consistency_losses_logged)
            else:
                consistency_losses_logged = 0.0

            # collecting here supervised loss per-epoch
            results.append(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if self.unsupervised_loss_weight > 0:
                summary_dict["consistency_loss"] = consistency_losses_logged
                summary_dict["unsup_weight"] = effective_unsup_weight
            stop_training = False
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                metric = val_metrics.get(self.early_stopping_monitor)
                if metric is not None:
                    if metric + self.early_stopping_min_delta < best_metric:
                        best_metric = metric
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                        if 0 < self.early_stopping_patience <= epochs_since_improvement:
                            stop_training = True
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)
            if stop_training:
                break
        return results
