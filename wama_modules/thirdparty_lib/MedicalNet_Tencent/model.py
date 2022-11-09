import torch
from torch import nn
from .models import resnet


def generate_model(model_depth):
    if model_depth == 10:
        model = resnet.resnet10()
    elif model_depth == 18:
        model = resnet.resnet18()
    elif model_depth == 34:
        model = resnet.resnet34()
    elif model_depth == 50:
        model = resnet.resnet50()
    elif model_depth == 101:
        model = resnet.resnet101()
    elif model_depth == 152:
        model = resnet.resnet152()
    elif model_depth == 200:
        model = resnet.resnet200()
    return model
