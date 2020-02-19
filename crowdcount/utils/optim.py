# -*-coding: utf-8-*-
import torch.nn as nn


def weights_normal_init(model, dev=0.01):
    """For modules in model,
    if module is Convolutional layer: init with torch.nn.init.normal_(mean, std),
    elif module is Linear layer: init with torch.nn.init.fill_(val),
    elif module is Batch Normalization layer: init with torch.nn.init.constant_(tensor, val)

    Args:
        model (torch.nn.module): the model to be init
        dev (float): the standard deviation used in norm init

    Returns:
        model (torch.nn.module)

    """
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    return model
