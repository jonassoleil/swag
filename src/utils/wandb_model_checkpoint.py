import wandb
import pytorch_lightning as pl

class WandBModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _save_model(self, trainer, filepath: str, pl_module=None):
        # monkey patch, support both versions
        if 'pl_module' in super()._save_model.__code__.co_varnames:
            super()._save_model(trainer, filepath)
        else:
            super()._save_model(trainer, filepath, pl_module=None)
        # TODO: make sure it works
        # print('saving for epoch: ', trainer.current_epoch)
        # print('saving: ', filepath)
        wandb.save(filepath, base_path=wandb.run.dir)