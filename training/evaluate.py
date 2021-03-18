"""Experiment-running framework."""
import argparse
import importlib
import os

import numpy as np
import torch
import pytorch_lightning as pl
import wandb
from tqdm import tqdm

from src import lit_models
from src.data.torchvision_dataset import TorchvisionDataset
from src.lit_models.lit_model import LitModel
from src.models.get_model import get_model
from src.util import filter_args_for_fn
from src.utils.load_utils import list_all_checkpoints, download_checkpoint
from src.utils.wandb_model_checkpoint import WandBModelCheckpoint

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
    # parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--mode", type=str, default="normal") # add swa, swag, ensemble

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    TorchvisionDataset.add_to_argparse(data_group)

    # model_group = parser.add_argument_group("Model Args")
    # model_class.add_to_argparse(model_group)
    parser.add_argument("--model_name", type=str, default="vgg16")
    parser.add_argument("--n_classes", type=int, default=10)

    parser.add_argument("--use_test", action="store_true", help="Use test set for evaluation")

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
    model = get_model(model_name=args.model_name, n_classes=args.n_classes, freeze=False, pretrained=True)

    checkpoints = list_all_checkpoints(args.run)
    print(checkpoints)
    download_checkpoint(args.run, 'last_checkpoints/last.ckpt') # TODO: get best instead?
    lit_model = LitModel.load_from_checkpoint('last_checkpoints/last.ckpt', args=vars(args), model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    data.prepare_data()
    data.setup()
    if args.use_test:
        dataloader = data.test_dataloader()
    else:
        dataloader = data.val_dataloader()

    predictions = []
    for idx, batch in tqdm(enumerate(dataloader)):
        pred = lit_model.test_step(batch, idx)
        pred = pred.cpu().numpy()
        predictions.append(pred)

    predictions = np.concatenate(predictions, axis=0)
    pred_path = os.path.join(wandb.run.dir, 'predictions.npy')
    np.save(pred_path, predictions)
    wandb.save(pred_path)



    # args.weights_summary = "full"  # Print full summary of the model
    # trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
