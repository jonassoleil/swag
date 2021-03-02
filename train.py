import models
from utils.get_datasets import get_datasets
from utils.experiment_utils import evaluate, train
from evaluators.evaluator import Evaluator
import torch
from modules.model_saver import ModelSaver
import os
import json

CONFIG = {
    "in_domain_dataset": "MNIST",
    # "ood_dataset": 'KMNIST',
    "pretrained": True,
    "freeze": True,
    "n_classes": 10,
    "n_epochs": 10,
    "model_name": "mnist_vgg16",
    "optimizer_name": "Adam",
    "optimizer_kwargs": {
        "lr": 3e-4,
    },
    "save_model": True,
    "scheduler": None,
    "scheduler_kwargs": {}
}

def get_valid_path(experiment_path):
    if os.path.exists(experiment_path):
        print(f'Experiment {experiment_path} already exists, will use {experiment_path+"_0"}')
        experiment_path += '_0'
        return get_valid_path(experiment_path) # check recursive
    else:
        return experiment_path

def run_training(
    experiment_path,
    in_domain_dataset,
    pretrained,
    freeze,
    n_classes,
    n_epochs,
    model_name,
    optimizer_name,
    optimizer_kwargs,
    save_model,
    scheduler,
    scheduler_kwargs
):
    # INIT
    experiment_path = get_valid_path(experiment_path)
    os.mkdir(experiment_path)
    config = locals()
    with open(os.path.join(experiment_path, 'config.json'), 'w') as fh:
        print(config)
        json.dump(config, fh)

    train_loader, test_loader = get_datasets(
        in_domain_dataset, batch_size_train=256, batch_size_test=1024
    )
    # TODO: some metrics and evaluation on OOD datasets
    # _, ood_test_loader = get_datasets(ood_dataset, batch_size_train=256, batch_size_test=1024)

    # model
    model_builder = getattr(models, model_name)
    model = model_builder(pretrained, n_classes, freeze)

    # init optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    loss = torch.nn.CrossEntropyLoss()

    # evaluators
    train_evaluator = Evaluator(n_classes, "TRAIN", path=experiment_path)
    trainval_evaluator = Evaluator(n_classes, "TRAINVAL", path=experiment_path)
    test_evaluator = Evaluator(n_classes, "TEST", path=experiment_path)
    saver = ModelSaver(experiment_path, mode='last')
    
    if scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler)
        scheduler = scheduler_class(optimizer, scheduler_kwargs)

    for i in range(n_epochs):
        # train
        train(i, train_loader, model, train_evaluator, optimizer, loss, scheduler)
        # trainval
        evaluate(i, train_loader, model, trainval_evaluator, loss)
        # test
        test_metrics = evaluate(i, test_loader, model, test_evaluator, loss) # this could be validation set
        
        # save (here will probably swag stuff go)
        if save_model:
            saver.save_if_needed(model, i, test_metrics) # TODO: don't do early stopping on test set

