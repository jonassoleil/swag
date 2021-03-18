"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from src import lit_models
from src.data.torchvision_dataset import TorchvisionDataset
from src.lit_models.lit_model import LitModel
from src.models.get_model import get_model
from src.util import filter_args_for_fn

wandb.init(project='swa', entity='adv-ml')

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    TorchvisionDataset.add_to_argparse(data_group)

    # model_group = parser.add_argument_group("Model Args")
    # model_class.add_to_argparse(model_group)


    parser.add_argument("--model_name", type=str, default="vgg16")
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--freeze", action='store_true')


    lit_model_group = parser.add_argument_group("LitModel Args")
    LitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data = TorchvisionDataset(args)
    model = get_model(**filter_args_for_fn(vars(args), get_model))

    if args.load_checkpoint is not None:
        lit_model = LitModel.load_from_checkpoint(args.load_checkpoint, args=vars(args), model=model)
    else:
        lit_model = LitModel(args=vars(args), model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=30)] # what if cyclical?

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs")

    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
