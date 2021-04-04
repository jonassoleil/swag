import torch

from src.data import BaseDataModule

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, shape, len=2000):
        self.shape = list(shape)
        self.len = len
        self.targets = torch.zeros(len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.randn(self.shape), 0

class NoiseDatasetPL(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)
        if  args.dataset_name == 'MNIST':
            self.shape = (28, 28, 1)
        else:
            self.shape = (32, 32, 3)

    def setup(self, stage=None):
        self.data_train = NoiseDataset(self.shape)
        self.data_val = NoiseDataset(self.shape)
        self.data_test = NoiseDataset(self.shape)

