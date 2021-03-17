from src.models.mnist_vgg16 import mnist_vgg16
from torchvision import models
import torch

def get_model(model_name, pretrained, n_classes, freeze):
    if model_name == 'mnist_vgg16':
        return mnist_vgg16(pretrained, n_classes, freeze)

    model_class = getattr(models, model_name)
    model = model_class(pretrained=pretrained)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # replace last layer (torchvision models are a but inconsistent wrt to that)
    if 'vgg' in model_name:
        last_in = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features=last_in, out_features=n_classes, bias=True)
    else:
        last_in = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=last_in, out_features=n_classes, bias=True)

    return model

def model_add_to_argparse(parser):
    # TODO: finish this
    return parser