import torch
# Not sure if this stuff works as is but it might

def to_gpu(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def get_bns(module):
  bns = []
  for name, layer in module.named_modules():
    if issubclass(layer.__class__, torch.nn.modules.batchnorm._BatchNorm):
      bns.append((name, layer, layer.momentum))
  return bns

def reset_momenta(bns):
  for _, layer, momentum in bns:
    layer.momentum = momentum

def set_momenta(momentum, bns):
  for _, layer, _ in bns:
    layer.momentum = momentum

def reset_bns(bns):
  for _, layer, _ in bns:
    layer.running_mean = to_gpu(torch.zeros_like(layer.running_mean))
    layer.running_var = to_gpu(torch.ones_like(layer.running_var))

def update_batch_normalization(model, loader):
    bns = get_bns(model)
    model.train()
    reset_bns(bns)
    n = 0
    for X, _ in loader:
      X = to_gpu(X)
      input_var = X
      # input_var = torch.autograd.Variable(X)
      b = input_var.data.size(0)

      momentum = b / (n + b)
      set_momenta(momentum, bns)

      model(to_gpu(input_var))
      n += b
    reset_momenta(bns)
    model.eval()