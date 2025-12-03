from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from hydra.utils import get_original_cwd
import hydra



class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        lambda_cps,
    ):
        self.device = device
        self.models = models
        self.lambda_cps = lambda_cps


        # Optim related things
        self.supervised_criterion = supervised_criterion
        #all_params = [p for m in self.models for p in m.parameters()]        
        #self.optimizer = optimizer(params=all_params)

        #split optimizers and schedulers for each model CPS
        self.optimizers = [optimizer(params=m.parameters()) for m in self.models]
        
        #self.scheduler = scheduler(optimizer=self.optimizer)
        self.schedulers = [scheduler(optimizer=opt) for opt in self.optimizers]

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.unlabeled_dataloader = datamodule.unsupervised_train_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # eval only one model like in paper
                
                preds = self.models[0](x)
                
                val_loss = torch.nn.functional.mse_loss(preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}
    
    def CPS_loss(self, preds_one, preds_two, lambda_cps):
        with torch.no_grad():
            pseudo_labels_one = torch.argmax(preds_one, dim=1)
            pseudo_labels_two = torch.argmax(preds_two, dim=1)
        loss_one = torch.nn.functional.cross_entropy(preds_one, pseudo_labels_two)
        loss_two = torch.nn.functional.cross_entropy(preds_two, pseudo_labels_one)
        return lambda_cps * (loss_one + loss_two)


    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        results = []
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            #training loop rewamped for CPS

            for (x_labeled, targets), (x_unlabeled, _) in zip(self.train_dataloader, self.unlabeled_dataloader):
                x_labeled, targets = x_labeled.to(self.device), targets.to(self.device)
                x_unlabeled = x_unlabeled.to(self.device)

                # Zero gradients for all optimizers
                for opt in self.optimizers:
                    opt.zero_grad()

                # Supervised loss
                preds_l = [model(x_labeled) for model in self.models]
                supervised_losses = [self.supervised_criterion(p, targets) for p in preds_l]

                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))
                # Semi-supervised CPS loss
                predsu = [model(x_unlabeled) for model in self.models]
                cps_unlabeled = self.CPS_loss(predsu[0], predsu[1], lambda_cps=self.lambda_cps)
                cps_labeled = self.CPS_loss(preds_l[0], preds_l[1], lambda_cps=self.lambda_cps)
                
                total_loss = supervised_loss + cps_unlabeled + cps_labeled

                total_loss.backward() 

                # Step all optimizers
                for opt in self.optimizers:
                    opt.step()

            # collect per-epoch supervised loss for return
            epoch_supervised_loss = float(np.mean(supervised_losses_logged))
            results.append(epoch_supervised_loss)
            summary_dict = {
                "supervised_loss": epoch_supervised_loss,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)
        
        #save model weights
        save_dir = Path(get_original_cwd()) / "models" /self.logger.run.id
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), save_dir / f"model_{i}.pt")
            print(f"Saved model {i} weights to {save_dir / f'model_{i}.pt'}")
            
        return results


