"""Experiment-running framework."""
import argparse
import importlib
import os

import numpy as np
import torch
import pytorch_lightning as pl
import wandb
from tqdm import tqdm

from src.data.torchvision_dataset import TorchvisionDataset
from src.lit_models.lit_model import LitModel
from src.models.get_model import get_model
from src.modules.dummy_iterator import DummyIterator
from src.modules.ensemble_iterator import EnsembleIterator
from src.modules.swa import apply_swa
from src.modules.swag_iterator import SWAGIterator
from src.utils.load_utils import list_all_checkpoints, download_checkpoint
from sklearn.metrics import accuracy_score
from src.utils.update_bn import update_batch_normalization

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

    # evaluation arguments
    parser.add_argument("--run", type=str, default=None, required=True, help='Run id from which to load the checkpoints')
    parser.add_argument("--checkpoint", type=str, default='last_checkpoints/last.ckpt',
                        help='Specific checkpoint to load (only for normal mode)')
    parser.add_argument("--mode", type=str, default="normal", help='Evaluation mode: normal|swa|swag|ensemble|(interpolate?)') #
    parser.add_argument("--k", type=int, default=None, help="Number of last checkpoints to use (if None all will be used)") # for swa, swag, ensemble
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples for SWAG and interpolate modes") # for SWAG

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

def evaluate_model(lit_model, dataloader):
    predictions = []
    for idx, batch in tqdm(enumerate(dataloader)):
        pred = lit_model.test_step(batch, idx)
        pred = pred.cpu().detach().numpy()
        predictions.append(pred)
    predictions = np.concatenate(predictions, axis=0)
    return predictions


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

    # prepare data
    data.prepare_data()
    data.setup()
    if args.use_test:
        targets = np.array(data.data_test.targets)
        dataloader = data.test_dataloader()
    else:
        targets = np.array(data.data_val.targets)
        dataloader = data.val_dataloader()

    # Prepare model
    model = get_model(model_name=args.model_name, n_classes=args.n_classes, freeze=False, pretrained=True)

    if args.mode == 'normal':
        download_checkpoint(args.run, args.checkpoint)
        lit_model = LitModel.load_from_checkpoint(args.checkpoint, args=vars(args), model=model)
        model_iterator = DummyIterator(lit_model)

    elif args.mode == 'swa':
        lit_model = LitModel(args=vars(args), model=model)
        apply_swa(lit_model, args.run, K=args.swa_k)
        update_batch_normalization(lit_model, data.train_dataloader()) #
        model_iterator = DummyIterator(lit_model)

    elif args.mode == 'ensemble':
        lit_model = LitModel(args=vars(args), model=model)
        model_iterator = EnsembleIterator(lit_model, args.run, K=args.k)

    elif args.mode == 'swag':
        lit_model = LitModel(args=vars(args), model=model)
        model_iterator = SWAGIterator(lit_model, args.run,  data.train_dataloader(), K=args.k, n_samples=args.n_samples)

    # TODO: some better logging
    # logger = pl.loggers.TensorBoardLogger("training/logs")
    # if args.wandb:
    #     logger = pl.loggers.WandbLogger()
    #     logger.watch(model)
    #     logger.log_hyperparams(vars(args))

    # Make predictions
    predictions = []
    for i, model in enumerate(model_iterator): # iterate over all model versions (for ensemble/swag)
        preds_single = evaluate_model(model, dataloader)
        predictions.append(preds_single)
        print(i, accuracy_score(targets, preds_single.argmax(axis=1)))
    predictions = np.concatenate(predictions, axis=0)

    # save targets and predictions
    pred_path = os.path.join(wandb.run.dir, 'predictions.npy')
    np.save(pred_path, predictions)
    wandb.save(pred_path)
    targets_path = os.path.join(wandb.run.dir, 'targets.npy')
    np.save(targets_path, targets)
    wandb.save(targets_path)

    print(accuracy_score(targets, predictions.mean(axis=0).argmax(axis=1)))



if __name__ == "__main__":
    main()
