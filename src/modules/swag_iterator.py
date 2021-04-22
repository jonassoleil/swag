import torch
import numpy as np
import gc
from src.modules.base_model_iterator import BaseModelIterator
from src.modules.interpolated_iterator import get_flattened_params, flattened_to_state_dict
from src.modules.swa import load_swa_weights
from src.utils.load_utils import get_k_last_checkpoints, get_state_from_checkpoint
from src.utils.update_bn import update_batch_normalization
import os, psutil
from collections import OrderedDict


def flattened_numpy_to_state_dict(flattened_params, shape_dict):
    new_state_dict = OrderedDict()
    i = 0
    for k, w_shape in shape_dict.items():
      w_size = np.prod(w_shape)
      new_state_dict[k] = torch.reshape(torch.from_numpy(flattened_params[int(i):int(i+w_size)]), w_shape)
      i += w_size
    return new_state_dict

class SWAGIterator(BaseModelIterator):
    """
    Iterate over SWAG weight samples
    """
    def __init__(self, model, run_id, train_loader, K=None, n_samples=16):
        super().__init__()
        process = psutil.Process(os.getpid())
        print('Memory at init [~MB]: ', process.memory_info().rss/1e6)
        print('Memory at init [%]: ', process.memory_percent())
        self.checkpoints = get_k_last_checkpoints(run_id, K)
        self.length = n_samples
        self.train_loader = train_loader
        self.model = model
        self.run_id = run_id

        self.K = len(self.checkpoints)
        self.shape_dict = OrderedDict()
        for k, v in model.state_dict().items():
          self.shape_dict[k] = v.shape

        # compute mean
        self.w_mean = get_flattened_params(load_swa_weights(run_id, self.checkpoints)).numpy()
        gc.collect()

        self.D = len(self.w_mean)
        print(self.K, self.D)

        # initialize, use half-precision to save up RAM
        self.sigma_diag = np.zeros(self.D, dtype=np.float16)
        self.d_hat = np.zeros((self.K, self.D), dtype=np.float16)

        for i, ch in enumerate(self.checkpoints):
            print(f'Loading: {ch}')
            w = get_state_from_checkpoint(run_id, ch)
            wf = get_flattened_params(w).numpy()
            del w # clear memory as soon as possible
            gc.collect()

            # diagonal part
            # TODO: in-place
            self.sigma_diag = ((i * self.sigma_diag) + np.square(wf))/(i+1)

            # get low rank part
            self.d_hat[i,:] = wf - self.w_mean

        # diagonal variance (and take square root)
        # TODO: maybe this could be in place
        self.sigma_diag = np.sqrt(np.abs(self.sigma_diag - np.square(self.w_mean).astype(np.float16)))
        # save constants for convenience
        self.c1 = 1 / np.sqrt(2)
        self.c2 = 1 / np.sqrt(2 * (self.K - 1))

        process = psutil.Process(os.getpid())
        print('Memory after init [~MB]: ', process.memory_info().rss/1e6)
        print('Memory after init [%]: ', process.memory_percent())

    def sample_weights(self):
        # from eq (1) in SWAG paper
        # z1 = np.random.randn(self.D)
        z2 = np.random.randn(self.K).astype(np.float16)
        new_weights = np.random.randn(self.D).astype(np.float32) # z1
        new_weights *= self.sigma_diag # in-place ?
        new_weights *= self.c1.astype(np.float32)
        new_weights += self.w_mean
        new_weights += self.c2 * (z2 @ self.d_hat) # I guess not in place
        return flattened_numpy_to_state_dict(new_weights,
                                       self.shape_dict)


    def get_next_model(self):
        weights = self.sample_weights()
        self.model.load_state_dict(weights)
        gc.collect()
        print('updating batch norm')
        update_batch_normalization(self.model, self.train_loader)
        return self.model
