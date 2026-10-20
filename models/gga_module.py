from typing import Any, Optional
import lightning.pytorch as pl
import torch
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os
import json

from .timm_model import get_timm_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from .first_order_score import FirstOrderScorer

class ImageClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="timm/vit_base_patch32_224.orig_in21k",
        num_classes=8,
        adaptation_mode="head_only",
        lora_r=16,
        lora_target_modules=None,
        lora_modules_to_save=None,
        lora_adapters_name=None,
        warmup_epochs=10,
        run_gga=True,
        gga_score_name="score",
    ):
        super().__init__()
        self.model = get_timm_model(
            model_name=model_name,
            num_classes=num_classes,
            adaptation_mode=adaptation_mode,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_modules_to_save=lora_modules_to_save,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.save_hyperparameters()

        self.lora_adapters_name = lora_adapters_name
        self.warmup_epochs = warmup_epochs
        self.run_gga = run_gga
        self.gga_score_name = gga_score_name
        print(f"[INFO] ℹ️  lora_adapters_name: {self.lora_adapters_name}")


    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
            # num_params += param.numel()
            print(f"{name}: {param.shape}; num_params: {param.numel()}; req_grad: {param.requires_grad}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        acc = self.train_acc(outputs, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_end(self):

        self.eval()
        # Define target modules
        target_modules = [
            f"blocks.{i}.{suffix}"
            for i in range(12)
            for suffix in ["attn", "mlp"]
        ]

        # Get the trainer's train dataloader
        train_loader = self.trainer.datamodule.train_dataloader()
        device = next(self.model.parameters()).device  # get model's device
        gga = FirstOrderScorer(self.model, target_modules)
        scores = gga.compute_scores(
            self.criterion,
            train_loader,
            device,
            num_samples=512
        )

        # dataset_name = self.trainer.datamodule.hparams["dataset_name"].split("/")[1]
        os.makedirs("gga_scores", exist_ok=True)
        save_path = os.path.join("gga_scores", f"{self.gga_score_name}.json")

        with open(save_path, "w") as f:
            json.dump(scores, f, indent=4)

        print(f"[INFO] ℹ️  GGA scores saved to {save_path}")

    def configure_optimizers(self):
        # Get optimizer class dynamically
        # optimizer_class = self.hparams.optimizer["class_path"]
        # optimizer_params = self.hparams.optimizer["init_args"]
        
        # Instantiate optimizer
        # optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4)
        
        if self.run_gga:
            return optimizer

        # Cosine LR Scheduler from timm
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.trainer.max_epochs - self.warmup_epochs,   # or max_steps, if step-based
            warmup_t=self.warmup_epochs,  # Warmup epochs
            warmup_lr_init=1e-5,
            lr_min=1e-6,
            cycle_limit=1,
            warmup_prefix=True,
            t_in_epochs=True  # If training is epoch-based
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Using epoch-based scheduling
                "frequency": 1,
                "name": "cosine_lr",  # Name for logging
                "monitor": "valid_acc_epoch",  # Optional: monitor metric for some schedulers
            },
        }

    def lr_scheduler_step(self, scheduler, metric: Optional[Any]):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
        # scheduler.

