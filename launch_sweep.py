import subprocess
import sys

import wandb

# Define your WandB sweep configuration
sweep_configuration = {
    "method": "random",
    "metric": {"name": "best_valid_acc_epoch", "goal": "maximize"},  # Example metric
    "parameters": {
        "trainer.max_epochs": {"values": [5, 10, 20, 25, 50, 100, 200]},
        "model.model_name": {
            "values": [
                # "timm/vit_base_patch14_reg4_dinov2.lvd142m",
                # "timm/vit_base_patch16_clip_224.openai",
                # "timm/vit_base_patch16_224.orig_in21k_ft_in1k",
                "timm/vit_base_patch16_224.orig_in21k",
                # "timm/samvit_base_patch16.sa1b",
                # "timm/hiera_base_224.mae",
                # "timm/hiera_base_224.mae_in1k_ft_in1k",
            ]
        },
        "model.adaptation_mode": {"values": ["full", "head_only", "lora"]},
        "model.lora_r": {"values": [8, 16, 32, 64]},
        "optimizer.class_path": {
            "values": ["torch.optim.AdamW"]
        },
        "optimizer.init_args.lr": {"values": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]},
        "optimizer.init_args.weight_decay": {"values": [0.0, 1e-6, 1e-4, 1e-2]},
        "data.batch_size": {"values": [32, 64, 128]},
        "trainer.accumulate_grad_batches": {"values": [1, 2, 4]},
        "data.train_crop_mode": {"values": ["rrc", "rkrc", "rkrr"]},
        "data.train_hflip": {"values": [0.0, 0.5]},
        "data.train_vflip": {"values": [0.0, 0.5]},
        "data.train_color_jitter_prob": {"values": [0.0, 0.25, 0.5]},
        "data.train_grayscale_prob": {"values": [0.0, 0.25, 0.5]},
        "data.train_gaussian_blur_prob": {"values": [0.0, 0.25, 0.5]},
        "data.eval_crop_pct": {"values": [0.8, 0.9, 0.95, 1.0]},
        "data.eval_crop_mode": {"values": ["border", "center", "squash"]},
    },
}


def cli_main():
    # Initialize a WandB sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Grid_Search")
    args = sys.argv[1:]

    # Define the function to run for each hyperparameter configuration
    def run():
        wandb.init(project="VTAB_adaptation")
        config = wandb.config

        config["data.model_name"] = config["model.model_name"]
        config["lr_scheduler.init_args.T_max"] = config["trainer.max_epochs"]
        config["trainer.enable_checkpointing"] = False

        args_used = args + [f"--{k}={v}" for k, v in config.items()]
        # Launch the worker script as a subprocess
        subprocess.run([sys.executable, "train_on_task.py"] + args_used)

    # Launch the sweep with the worker function
    wandb.agent(sweep_id, function=run, count=500)


if __name__ == "__main__":
    cli_main()
