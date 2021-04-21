import torch
import numpy as np

from src.modules.base_model_iterator import BaseModelIterator
from src.modules.interpolated_iterator import get_flattened_params, flattened_to_state_dict
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

        checkpoint_weights_flattened = []
        for ch in self.checkpoints:
            w = get_state_from_checkpoint(run_id, ch)
            wf = get_flattened_params(w)
            checkpoint_weights_flattened.append(wf)

        d = torch.stack(checkpoint_weights_flattened)
        # compute mean
        self.w_mean = torch.mean(d, dim=0)
        self.D = len(self.w_mean)
        # compute diagonal variance (and take square root)
        self.sigma_diag = torch.sqrt(torch.abs(torch.mean(torch.square(d), dim=0) - torch.square(self.w_mean)))
        # get low rank part
        self.d_hat = d - self.w_mean
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
