import torch
from torchvision import models

# this is totally arbitrary, just wanted something there
def mnist_vgg16(pretrained, n_classes, freeze):

  model = models.vgg16(pretrained=pretrained)

  if freeze:
    for param in model.parameters():
      param.requires_grad = False

  # replace classifier
  model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=6272, out_features=4096, bias=True),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(p=0.5, inplace=False),
    torch.nn.Linear(in_features=4096, out_features=n_classes, bias=True),
  )

  # replace input
  model.features = model.features[:10]
  model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  return model