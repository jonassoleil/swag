import argparse
import pytorch_lightning as pl
import torch
from src.util import get_kwargs_by_prefix

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"

class LitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """
    def __init__(self, model, args: dict=None):
        super().__init__()
        self.model = model
        self.args = args if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.optimizer_kwargs = get_kwargs_by_prefix(self.args, 'optimizer_')

        self.scheduler_name = args.get('scheduler')
        self.scheduler_kwargs = get_kwargs_by_prefix(self.args, 'scheduler_')

        loss = self.args.get("loss", LOSS)
        if not loss in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--optimizer_lr", type=float, default=LR)
        parser.add_argument("--optimizer_momentum", type=float, default=None) # won't be used if None
        parser.add_argument("--optimizer_weight_decay", type=float, default=None) # won't be used if None
        
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")

        parser.add_argument("--scheduler", type=str, default=None, 
                help="scheduler CosineAnnealingWarmRestarts or LambdaLR (or nothing)")
        parser.add_argument("--scheduler_cycle_length", type=int, default=10, 
                help="scheduler cycle length in epochs")
        parser.add_argument("--scheduler_min_lr", type=float, default=1e-6, 
                help="scheduler min lr")

        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        if self.scheduler_name is None:
            return optimizer
        scheduler = get_scheduler(self.scheduler_name, optimizer, **self.scheduler_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(probs, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(probs, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return logits

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        self.test_acc(probs, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        return logits


def get_schedule_fn(cycle_size, min_fraction=0.1):
  def _schedule_fn(step):
    return 1 - (1 - min_fraction) * ((step % cycle_size) / cycle_size)
  return _schedule_fn

def get_scheduler(name, optimizer, cycle_length=1, min_lr=1e-6):
    # calculate length of a cycle for the given dataset
    if name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_length, T_mult=1, eta_min=min_lr)
    elif name == 'LambdaLR':
        lr_fraction = min_lr / optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_schedule_fn(cycle_length, lr_fraction))
    else:
        raise ValueError(f'Scheduler {name} is not supported')