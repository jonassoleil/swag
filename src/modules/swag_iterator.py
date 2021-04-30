import torch
import numpy as np

from src.modules.base_model_iterator import BaseModelIterator
from src.modules.interpolated_iterator import get_flattened_params, flattened_to_state_dict
from src.modules.swa import load_swa_weights
from src.utils.load_utils import get_k_last_checkpoints, get_state_from_checkpoint
from src.utils.update_bn import update_batch_normalization


class SWAGIterator(BaseModelIterator):
    """
    Iterate over SWAG weight samples
    """
    def __init__(self, model, run_id, train_loader, K=None, n_samples=16):
        super().__init__()
        self.checkpoints = get_k_last_checkpoints(run_id, K)
        self.length = n_samples
        self.train_loader = train_loader
        self.model = model
        self.run_id = run_id

        self.K = len(self.checkpoints)

        # compute mean
        self.w_mean = get_flattened_params(load_swa_weights(run_id, self.checkpoints))

        self.D = len(self.w_mean)
        print(self.K, self.D)

        # initialize
        self.sigma_diag = torch.zeros(self.D)
        self.d_hat = torch.zeros((self.K, self.D))

        for i, ch in enumerate(self.checkpoints):
            print(f'Loading: {ch}')
            w = get_state_from_checkpoint(run_id, ch)
            wf = get_flattened_params(w)

            # diagonal part
            self.sigma_diag = ((i * self.sigma_diag) + torch.square(wf))/(i+1)

            # get low rank part
            self.d_hat[i,:] = wf - self.w_mean

        # diagonal variance (and take square root)
        self.sigma_diag = torch.sqrt(torch.abs(self.sigma_diag - torch.square(self.w_mean)))
        # save constants for convenience
        self.c1 = 1 / np.sqrt(2)
        self.c2 = 1 / np.sqrt(2 * (self.K - 1))

    def sample_weights(self):
        # from eq (1) in SWAG paper
        z1 = torch.randn(self.D)
        z2 = torch.randn(self.K)
        new_weights = self.w_mean + self.c1 * z1 * self.sigma_diag + self.c2 * (z2 @ self.d_hat)

        return flattened_to_state_dict(new_weights, self.model.state_dict())


    def get_next_model(self):
        weights = self.sample_weights()
        self.model.load_state_dict(weights)
        print('updating batch norm')
        update_batch_normalization(self.model, self.train_loader)
        return self.model
