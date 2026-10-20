import argparse
import importlib
import sys
import torch

from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI
from data.vtab_datamodule import VTABDataModule
from callbacks.save_config import LoggerSaveConfigCallback

MODEL_MODULES = {
    "base": "models.image_classification_module",
    "lora-drop": "models.lora_drop_module",
    "gga": "models.gga_module",
}

def cli_main():
    """
    Main function to run the CLI for training an image classification model on a specific task.
    """
    # Check GPU availability
    if not torch.cuda.is_available():
        print(f"[ERROR] No GPU available. Exiting ...")
        sys.exit(1) # Exit the script with error code 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_everything", 
        type=int,
        default=42, 
        help="Seed for reproducibility"
    )
    parser.add_argument(
        "--adaptation-mode",
        choices=MODEL_MODULES.keys(),
        default="base",
    )
    args, _ = parser.parse_known_args()

    seed_everything(args.seed_everything, workers=True)
    print(f"[INFO] Seed set to {args.seed_everything} with workers=True")

    module = importlib.import_module(
        MODEL_MODULES[args.adaptation_mode]
    )

    sys.argv = [arg for arg in sys.argv if not arg.startswith("--adaptation-mode")]
    print(sys.argv)

    ModelClass = module.ImageClassificationLightningModule

    print(f"[INFO] Using model class: {ModelClass}")

    cli = LightningCLI(
        ModelClass,
        VTABDataModule,
        save_config_callback=LoggerSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()
