from collections import OrderedDict

import torch
import numpy as np
from src.modules.base_model_iterator import BaseModelIterator
from src.utils.load_utils import get_k_last_checkpoints, get_state_from_checkpoint
from src.utils.update_bn import update_batch_normalization


def get_flattened_params(state_dict):
  flattened_params = []
  for _, weight in state_dict.items():
    flattened_params.append(torch.flatten(weight))
  flattened_params = torch.cat(flattened_params)
  return flattened_params

def flattened_to_state_dict(flattened_params, state_dict):
    new_state_dict = OrderedDict()
    i = 0
    for k, v in state_dict.items():
      w_shape = v.shape
      w_size = np.prod(w_shape)
      new_state_dict[k] = torch.reshape(flattened_params[int(i):int(i+w_size)], w_shape)
      i += w_size
    return new_state_dict

def get_point_along_axis(w0, w1, loc):
    d = w1 - w0
    abs_dist = loc * torch.norm(d)
    w = w0 + loc * d
    return w, abs_dist

class InterpolatedIterator(BaseModelIterator):
    """
    Iterated over models in an ensemble
    """
    def __init__(self, model, run_id, train_loader, epochs=None, n_samples=16):
        super().__init__()
        if epochs == None:
            # get two last checkpoints
            self.checkpoints = get_k_last_checkpoints(run_id, 2)
        else:
            raise NotImplementedError
        self.length = n_samples
        self.model = model
        self.train_loader = train_loader
        self.run_id = run_id
        w0 = get_state_from_checkpoint(run_id, self.checkpoints[0])
        w1 = get_state_from_checkpoint(run_id, self.checkpoints[1])
        self.w0f = get_flattened_params(w0)
        self.w1f = get_flattened_params(w1)
        self.relative_locations = np.arange(-1/(n_samples-2), 1 + 2/(n_samples-2), 1/(n_samples-2))

    def get_next_model(self):
        print('loc:', self.relative_locations[self.i])
        wif, abs_dist = get_point_along_axis(self.w0f, self.w1f, loc=self.relative_locations[self.i])
        weights = flattened_to_state_dict(wif, self.model.state_dict())
        self.model.load_state_dict(weights)
        update_batch_normalization(self.model, self.train_loader)
        return self.model


