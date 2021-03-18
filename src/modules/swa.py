import torch
import os

# TODO: make it work
# def load_swa_weights(path, epochs):
#     n = len(epochs)
#     # load first epoch
#     weights = torch.load(os.path.join(path, f'model_{epochs[0]}.pth'))
#
#     # shrink
#     for k, w in weights.items():
#         weights[k] = w / n
#
#     # compute average
#     for i in epochs[1:]:
#         x = torch.load(os.path.join(path, f'model_{i}.pth'))
#         for k, w in x.items():
#             weights[k] += w / n
#
#     return weights
#
#
# def apply_swa(model, path, K=None):
#     available_epochs = list(sorted(get_available_epochs(path)))
#     # None for all epochs
#     if K is None:
#         epochs = available_epochs
#     else:
#         if K > len(available_epochs):
#             raise ValueError(f'K={K} is larger than the number of available epochs: {available_epochs}')
#         epochs = available_epochs[-K:]
#     weights = load_swa_weights(path, epochs)
#     model.load_state_dict(weights)
#     # to_gpu(model)

