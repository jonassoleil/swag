import argparse

import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from torchvision import transforms

from src.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "download"


class CIFAR10(BaseDataModule):
    """
    CIFAR10 DataModule
    """

    def __init__(self, args: argparse.Namespace=None) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dims = (3, 32, 32)
        self.output_dims = (1,)
        self.mapping = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __repr__(self):
        std = f"CIFAR10 Dataset\nNum classes {len(self.mapping)}\n Mapping: {self.mapping}\n Dims: {self.dims} \n" 
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        
        a, b = next(iter(self.train_dataloader()))
        a, b = a.float(), b.float()
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch a stats: {(a.shape, a.dtype, a.min(), a.mean(), a.std(), a.max())}\n"
            f"Batch b stats: {(b.shape, b.dtype, b.min(), b.mean(), b.std(), b.max())}\n"
        )
        return std + data

    def prepare_data(self):
        """Donload train and test CIFAR10 data from Pytorch canonical source"""
        TorchCIFAR10(self.data_dir, train=True, download=True)
        TorchCIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Split into train, val, test and set dims.
        CIFAR has 6000 pr class, 50000 training and 
        10000 test images
        """
        cifar10_full = TorchCIFAR10(self.data_dir, train=True, transform=self.transform)
        print(len(cifar10_full))
        self.data_train, self.data_val = random_split(cifar10_full, [40000,10000])
        self.data_test = TorchCIFAR10(self.data_dir, train=False, transform=self.transform)

    if __name__ == "__main__":
        load_and_print_info(CIFAR10)
