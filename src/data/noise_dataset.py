import torch

from src.data import BaseDataModule

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, shape, len=2000, seed=0):
        self.shape = list(shape)
        self.len = len
        self.targets = torch.zeros(len)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.data = torch.randn([self.len] + self.shape)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], 0

class NoiseDatasetPL(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)
        if  args.dataset_name == 'MNIST':
            self.shape = (1, 28, 28)
        else:
            self.shape = (3, 32, 32)

    def setup(self, stage=None):
        self.data_train = NoiseDataset(self.shape)
        self.data_val = self.data_train
        self.data_test = self.data_train

