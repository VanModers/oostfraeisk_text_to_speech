"""Callbacks used by train_piper.py with Piper's manual optimization loop."""

from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer


class ManualLRSchedulerStep(Callback):
    """Step Piper's native schedulers once per epoch and log both learning rates."""

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        for name, optimizer in zip(("lr_g", "lr_d"), optimizers):
            raw_optimizer: Any = getattr(optimizer, "optimizer", optimizer)
            pl_module.log(
                name,
                raw_optimizer.param_groups[0]["lr"],
                on_step=False,
                on_epoch=True,
            )

        schedulers = pl_module.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()
