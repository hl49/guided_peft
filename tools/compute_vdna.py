import argparse
import lightning.pytorch as pl
import numpy as np
import timm
from datasets import load_dataset
from typing import Optional, Tuple, Union
from vdna import VDNAProcessor, EMD, load_vdna_from_files


VTAB_DATASETS = [
        "caltech101",
        "cifar100",
        "dtd",
        "flowers102",
        "pets",
        "sun397",
        "svhn",
        "eurosat",
        "resisc45",
        "patch_camelyon",
        "retinopathy",
        "clevr_count",
        "clevr_distance",
        "dsprites_location",
        "dsprites_orientation",
        "smallnorb_azimuth",
        "smallnorb_elevation",
        "dmlab",
        "kitti_distance",
    ]


class VTABDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name="username/vtab_dataset",  # Used to load the dataset splits
        model_name="timm/vit_base_patch16_224.orig_in21k",  # Used to get model transforms expected
        batch_size=128,
        eval_batch_size=8,
        num_workers: int = 4,
        train_split_name="train800",
        val_split_name="val200",
        test_split_name="test",
        train_no_aug: Optional[bool] = True,
        train_crop_mode: Optional[str] = None,
        train_scale: Optional[Tuple[float, float]] = None,
        train_ratio: Optional[Tuple[float, float]] = None,
        train_hflip: float = 0.0,
        train_vflip: float = 0.0,
        train_color_jitter: Union[float, Tuple[float, ...]] = 0.0,
        train_color_jitter_prob: Optional[float] = None,
        train_grayscale_prob: float = 0.0,
        train_gaussian_blur_prob: float = 0.0,
        eval_crop_pct: Optional[float] = None,
        eval_crop_mode: Optional[str] = None,
        eval_crop_border_pixels: Optional[int] = None,
        normalize: Optional[bool] = False,
    ):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage=None):
        model = timm.create_model(self.hparams.model_name, pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)

        print(f"[INFO] 📀 Dataset: {self.hparams.dataset_name}")
        print(f"[INFO] 👺 No-Augmentation: {self.hparams.train_no_aug}")
        print(f"[INFO] 📏 Mean: {data_config['mean']}")
        print(f"[INFO] 📏 STD: {data_config['std']}")
        print(f"[INFO] 📏 Normalize: {self.hparams.normalize}")

        data_config["normalize"] = self.hparams.normalize

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


def compute_imagenet_vdna(imagenet_path: str):
    vdna_proc = VDNAProcessor()

    vdna = vdna_proc.make_vdna(
        source=imagenet_path,
        feat_extractor_name="timm_vit_base",
        crop_to_square_pre_resize="center",
        num_workers=8,
    )

    vdna.save("./vdnas/vdna_imagenet_mini_timm_vit")


def compute_vtab_vdnas():

    vdna_proc = VDNAProcessor()

    for dataset in VTAB_DATASETS:
        print(f"Processing {dataset}")

        dm = VTABDataModule(
            dataset_name=f"username/vtab_{dataset}"
        )
        dm.setup()

        images_list = [
            np.array(sample["image"])
            for sample in dm.ds_train
        ]

        vdna = vdna_proc.make_vdna(
            source=images_list,
            feat_extractor_name="timm_vit_base",
            crop_to_square_pre_resize="center",
            num_workers=8,
        )

        vdna.save(f"./vdnas/vdna_{dataset}_timm_vit")

def compute_emd():

    emd_in1k_vtab = []

    vdna_in1k = load_vdna_from_files(
        "./vdnas/vdna_imagenet_1k_timm_vit",
        device="cpu",
    )

    for dataset in VTAB_DATASETS:
        print(f"Computing EMD for {dataset}")

        vdna_vtab = load_vdna_from_files(
            f"./vdnas/vdna_{dataset}_timm_vit",
            device="cpu",
        )

        emd_score = EMD(vdna_in1k, vdna_vtab)
        emd_in1k_vtab.append(emd_score)

    np.save("emd_in1k_vtab.npy", emd_in1k_vtab)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        choices=["vtab", "imagenet-mini", "emd"],
        help="Dataset source to process",
    )
    parser.add_argument(
        "--imagenet-path",
        type=str,
        help="Path to ImageNet-Mini train directory",
    )

    args = parser.parse_args()

    if args.source == "imagenet-mini":
        if args.imagenet_path is None:
            parser.error(
                "--imagenet-path is required when source=imagenet-mini"
            )

        compute_imagenet_vdna(args.imagenet_path)

    elif args.source == "vtab":
        compute_vtab_vdnas()
    
    elif args.source == "emd":
        compute_emd()


if __name__ == "__main__":
    main()