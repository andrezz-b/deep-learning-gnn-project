from typing import Callable

import numpy as np
import torch
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

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
        unsupervised_warmup_epochs: int = 0,
        early_stopping: dict | None = None,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
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
        self.unsupervised_warmup_epochs = unsupervised_warmup_epochs
        self.early_stopping_cfg = early_stopping or {}
        self.early_stopping_monitor = self.early_stopping_cfg.get("monitor", "val_MSE")
        self.early_stopping_patience = self.early_stopping_cfg.get("patience", 0)
        self.early_stopping_min_delta = self.early_stopping_cfg.get("min_delta", 0.0)

    def _forward(self, model, batch, prefer_teacher: bool = False):
        if prefer_teacher:
            use_teacher = (
                getattr(model, "teacher", None) is not None
            )  # checks if the model has an attribute called teacher
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

                # Ensemble prediction
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
        # self.logger.log_dict()
        results = []
        best_metric = float("inf")
        epochs_since_improvement = 0

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            warmup_scale = 1.0
            if self.unsupervised_warmup_epochs > 0:
                warmup_scale = min(1.0, epoch / self.unsupervised_warmup_epochs)
            effective_unsup_weight = self.unsupervised_loss_weight * warmup_scale
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
                # Supervised loss
                supervised_losses = [
                    self.supervised_criterion(self._forward(model, x), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(
                    supervised_loss.detach().item() / len(self.models)
                )  # type: ignore
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
                loss.backward()  # type: ignore
                self.optimizer.step()
                self._update_mean_teachers()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            if consistency_losses_logged:
                consistency_losses_logged = np.mean(consistency_losses_logged)
            else:
                consistency_losses_logged = 0.0

            # collect per-epoch supervised loss for return
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
