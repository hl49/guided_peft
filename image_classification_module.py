from typing import Any, Optional
import lightning.pytorch as pl
import torch
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from .timm_model import get_timm_model
from timm.scheduler.cosine_lr import CosineLRScheduler


class ImageClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="timm/vit_base_patch32_224.orig_in21k",
        num_classes=8,
        adaptation_mode="full",
        lora_r=16,
        lora_target_modules=None,
        lora_modules_to_save=None,
        lora_adapters_name=None,
        warmup_epochs=10,
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
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.best_val_acc = 0.0
        self.save_hyperparameters()

        self.lora_adapters_name = lora_adapters_name
        self.warmup_epochs = warmup_epochs
        print(f"[INFO] ℹ️  lora_adapters_name: {self.lora_adapters_name}")


    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
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

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        self.val_acc.update(outputs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

            if self.lora_adapters_name:
                self.model.save_pretrained(self.lora_adapters_name)
                head_path=f"{self.lora_adapters_name}/head.pth"
                torch.save(self.model.head.state_dict(), head_path)

        self.log("valid_acc_epoch", val_acc)
        self.log("best_valid_acc_epoch", self.best_val_acc)
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4)

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