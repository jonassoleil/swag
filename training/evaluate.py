"""Experiment-running framework."""
import argparse
import importlib
import os
import shutil

import numpy as np
import torch
import pytorch_lightning as pl
import wandb
from tqdm import tqdm

from src.data.noise_dataset import NoiseDatasetPL
from src.data.torchvision_dataset import TorchvisionDataset, get_targets
from src.lit_models.lit_model import LitModel
from src.models.get_model import get_model
from src.modules.dummy_iterator import DummyIterator
from src.modules.ensemble_iterator import EnsembleIterator
from src.modules.swa import apply_swa
from src.modules.swag_iterator import SWAGIterator
from src.utils.load_utils import list_all_checkpoints, download_checkpoint, get_k_last_checkpoints
from sklearn.metrics import accuracy_score
from src.utils.update_bn import update_batch_normalization, to_gpu

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
    parser.add_argument("--mode", type=str, default="normal", help='Evaluation mode: normal|swa|swa_multiple|swag|swag_multiple|ensemble|(interpolate?)') #
    parser.add_argument("--evaluation_dataset_name", type=str, default=None, help='If a different evaluation dataset should be used.') #
    parser.add_argument("--k", type=int, default=None, help="Number of last checkpoints to use (if None all will be used)") # for swa, swag, ensemble
    parser.add_argument("--k_min", type=int, default=2, help="For swag_multiple") # for swa, swag, ensemble
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
    lit_model.eval()
    predictions = []
    for idx, batch in tqdm(enumerate(dataloader)):
        batch = [to_gpu(x) for x in batch]
        pred = lit_model.test_step(batch, idx)
        pred = pred.cpu().detach().numpy()
        predictions.append(pred)
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def run_evaluation(model_iterator, dataloader, targets, suffix=""):
    predictions = []
    for i, model in enumerate(model_iterator):  # iterate over all model versions (for ensemble/swag)
        preds_single = evaluate_model(model, dataloader)
        predictions.append(preds_single)
        print(i, accuracy_score(targets, preds_single.argmax(axis=1)))
    predictions = np.stack(predictions)

    # save targets and predictions
    pred_path = os.path.join(wandb.run.dir, f'predictions{suffix}.npy')
    np.save(pred_path, predictions)
    wandb.save(pred_path)
    targets_path = os.path.join(wandb.run.dir, f'targets.npy')
    np.save(targets_path, targets)
    wandb.save(targets_path)

    print(accuracy_score(targets, predictions.mean(axis=0).argmax(axis=1)))

def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    for checkpoint_path in ['last_checkpoints', 'cyclical_checkpoints']:
        if os.path.exists(checkpoint_path):
            print(f'removing: {checkpoint_path}')
            shutil.rmtree(checkpoint_path)

    parser = _setup_parser()
    args = parser.parse_args()
    data = TorchvisionDataset(args)
    if args.evaluation_dataset_name is not None:
        if args.evaluation_dataset_name == 'noise':
            evaluation_data = NoiseDatasetPL(args)
        else:
            eval_args = vars(args)
            eval_args['dataset_name'] = args.evaluation_dataset_name
            evaluation_data = TorchvisionDataset(argparse.Namespace(**eval_args))
        if args.mode in ["swa", "swa_multiple", "swag", "swag_multiple", "interpolate"]:
            data.prepare_data()
            data.setup()
    else:
        evaluation_data = data

    # prepare data
    evaluation_data.prepare_data()
    evaluation_data.setup()

    if args.use_test:
        targets = get_targets(evaluation_data.data_test)
        dataloader = evaluation_data.test_dataloader()
    else:
        targets = get_targets(evaluation_data.data_val)
        dataloader = evaluation_data.val_dataloader()

    # Prepare model
    model = get_model(model_name=args.model_name, n_classes=args.n_classes, freeze=False, pretrained=False)

    if args.mode == 'normal':
        download_checkpoint(args.run, args.checkpoint)
        lit_model = LitModel.load_from_checkpoint(args.checkpoint, args=vars(args), model=model)
        to_gpu(lit_model)
        to_gpu(lit_model.model)
        model_iterator = DummyIterator(lit_model)

    elif args.mode == 'swa':
        lit_model = LitModel(args=vars(args), model=model)
        apply_swa(lit_model, args.run, K=args.k)
        to_gpu(lit_model)
        to_gpu(lit_model.model)
        print('updating batch norm')
        update_batch_normalization(lit_model, data.train_dataloader()) #
        model_iterator = DummyIterator(lit_model)

    elif args.mode == 'ensemble':
        lit_model = LitModel(args=vars(args), model=model)
        to_gpu(lit_model)
        to_gpu(lit_model.model)
        model_iterator = EnsembleIterator(lit_model, args.run, K=args.k)

    elif args.mode == 'swag':
        lit_model = LitModel(args=vars(args), model=model)
        to_gpu(lit_model)
        to_gpu(lit_model.model)
        model_iterator = SWAGIterator(lit_model, args.run,  data.train_dataloader(), K=args.k, n_samples=args.n_samples)

    elif args.mode == 'swag_multiple':
        lit_model = LitModel(args=vars(args), model=model)
        to_gpu(lit_model)
        to_gpu(lit_model.model)
        if args.k is None:
            max_k = len(get_k_last_checkpoints(args.run))
        else:
            max_k = args.k

        for k in range(args.k_min, max_k + 1):
            print(f"running for K={k}")
            model_iterator = SWAGIterator(lit_model, args.run,
                                          data.train_dataloader(),
                                          K=k,
                                          n_samples=args.n_samples)
            run_evaluation(model_iterator, dataloader, targets, suffix=f'_k{k}')
        return

    elif args.mode == 'swa_multiple':
        lit_model = LitModel(args=vars(args), model=model)
        to_gpu(lit_model)
        to_gpu(lit_model.model)
        if args.k is None:
            max_k = len(get_k_last_checkpoints(args.run))
        else:
            max_k = args.k

        for k in range(args.k_min, max_k + 1):
            print(f"running for K={k}")
            apply_swa(lit_model, args.run, K=k)
            print('updating batch norm')
            update_batch_normalization(lit_model, data.train_dataloader())  #
            model_iterator = DummyIterator(lit_model)
            run_evaluation(model_iterator, dataloader, targets, suffix=f'_k{k}')
        return

    elif args.mode == 'interpolate':
        raise NotImplementedError

    else:
        raise ValueError(f'Mode {args.mode} is not available')

    run_evaluation(model_iterator, dataloader, targets)


if __name__ == "__main__":
    main()
