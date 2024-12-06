import importlib
import torch.nn as nn


def get_backbone(backbone, **kwargs):
    if backbone in ['vgg16', 'vgg19']:
        pretrained = getattr(importlib.import_module('torchvision.models'), backbone)(weights='DEFAULT').features
        return nn.Sequential(*list(pretrained.children())[:-1])
    elif backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        pretrained = getattr(importlib.import_module('torchvision.models'), backbone)(weights='DEFAULT')
        for param in pretrained.parameters():
            param.requires_grad = False
        for param in pretrained.layer4.parameters():
            param.requires_grad = True
        
        return nn.Sequential(*list(pretrained.children())[:-1])
