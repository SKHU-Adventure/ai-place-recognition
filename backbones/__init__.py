import importlib
import torch.nn as nn
from setup import config

def get_backbone(backbone, **kwargs):
    if backbone in ['vgg16', 'vgg19']:
        pretrained = getattr(importlib.import_module('torchvision.models'), backbone)(weights='DEFAULT').features

        for param in pretrained.parameters():
            param.requires_grad = False

    elif backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        pretrained = getattr(importlib.import_module('torchvision.models'), backbone)(weights='DEFAULT')
        
        for param in pretrained.parameters():
            param.requires_grad = False
        
        if config.finetune == 1:
            for param in pretrained.layer4.parameters():
                param.requires_grad = True

    return nn.Sequential(*list(pretrained.children())[:-1])
