import os
from argparse import Namespace

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.cli import SaveConfigCallback


class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, save_to_log_dir=False, **kwargs)

    def save_config(self, trainer, pl_module, stage: str) -> None:

        # Save the config file yaml in the log dir
        save_dir = trainer.loggers[0].save_dir  # this broadcasts the directory
        name = trainer.loggers[0].name
        version = trainer.loggers[0].version
        version = version if isinstance(version, str) else f"version_{version}"
        save_dir = os.path.join(save_dir, str(name), version, "config")
        assert save_dir is not None
        config_path = os.path.join(save_dir, self.config_filename)
        fs = get_filesystem(save_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(save_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

        # Log all hyperparams in the logger
        params = vars(self.config)

        params["effective_batch_size"] = (
            self.config.data.batch_size * self.config.trainer.accumulate_grad_batches
        )

        def log_recursive(logger, params, prefix=""):
            for k, v in params.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, Namespace):
                    log_recursive(logger, vars(v), prefix=full_key)
                else:
                    logger.log_hyperparams({full_key: v})

        for logger in trainer.loggers:
            log_recursive(logger, params)
