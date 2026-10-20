from typing import Sequence

import lightning.pytorch as pl
import timm
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Optional, Tuple, Union


class VTABDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name="clip-benchmark/wds_vtab-clevr_count_all",
        model_name="timm/vit_base_patch32_224.orig_in21k",  # Used to get model transforms expected
        batch_size=32,
        eval_batch_size=256,
        num_workers: int = 4,
        train_split_name:str ="train800",
        val_split_name:str = "val200",
        test_split_name: str = "test",
        train_no_aug: Optional[bool] = True,
        train_crop_mode: Optional[str] = None,
        train_scale: Optional[Tuple[float, float]] = None,
        train_ratio: Optional[Tuple[float, float]] = None,
        train_hflip: float = 0.5,
        train_vflip: float = 0.0,
        train_color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        train_color_jitter_prob: Optional[float] = None,
        train_grayscale_prob: float = 0.0,
        train_gaussian_blur_prob: float = 0.0,
        eval_crop_pct: Optional[float] = None,
        eval_crop_mode: Optional[str] = None,
        eval_crop_border_pixels: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage=None):
        model = timm.create_model(self.hparams.model_name, pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)

        data_config["no_aug"] = self.hparams.train_no_aug
        data_config["train_crop_mode"] = self.hparams.train_crop_mode
        data_config["scale"] = self.hparams.train_scale
        data_config["ratio"] = self.hparams.train_ratio
        data_config["hflip"] = self.hparams.train_hflip
        data_config["vflip"] = self.hparams.train_vflip
        data_config["color_jitter"] = self.hparams.train_color_jitter
        data_config["color_jitter_prob"] = self.hparams.train_color_jitter_prob
        data_config["grayscale_prob"] = self.hparams.train_grayscale_prob
        data_config["gaussian_blur_prob"] = self.hparams.train_gaussian_blur_prob

        data_config["crop_pct"] = self.hparams.eval_crop_pct
        data_config["crop_mode"] = self.hparams.eval_crop_mode
        data_config["crop_border_pixels"] = self.hparams.eval_crop_border_pixels

        self.transform_train = timm.data.create_transform(
            **data_config,
            is_training=True,
        )
        self.transform_eval = timm.data.create_transform(
            **data_config,
            is_training=False,
        )

        # Load the dataset splits
        self.ds_train = load_dataset(
            self.hparams.dataset_name, split=self.hparams.train_split_name
        )
        self.ds_valid = load_dataset(
            self.hparams.dataset_name, split=self.hparams.val_split_name
        )
        self.ds_test = load_dataset(
            self.hparams.dataset_name, split=self.hparams.test_split_name
        )

        # Preprocessing functions
        def process_train(batch):
            out_batch = {}
            out_batch["x"] = torch.stack(
                [self.transform_train(img.convert("RGB")) for img in batch["image"]]
            )
            out_batch["y"] = torch.tensor(batch["label"])
            return out_batch

        def process_eval(batch):
            out_batch = {}
            out_batch["x"] = torch.stack(
                [self.transform_eval(img.convert("RGB")) for img in batch["image"]]
            )
            out_batch["y"] = torch.tensor(batch["label"])
            return out_batch

        # Apply the preprocessing
        self.ds_train.set_transform(process_train)
        self.ds_valid.set_transform(process_eval)
        self.ds_test.set_transform(process_eval)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
