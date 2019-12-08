# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from .network import ConvUnit


class MCNN(nn.Module):
    """Refer from `"MCNN..." <https://www.semanticscholar.org/paper/Single-Image-Crowd-Counting-via-Multi-Column-Neural-Zhang-Zhou/2dc3b3eff8ded8914c8b536d05ee713ff0cdf3cd>`_ paper.

    Args:
        bn (bool): if True, Batch Normalization layer will be used after every convolutional layer

    """
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        print("*****init MCNN net*****")
        self.column1 = nn.Sequential(
            ConvUnit(3, 16, 9, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(16, 32, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(32, 16, 7, bn=bn),
            ConvUnit(16, 8, 7, bn=bn),
        )

        self.column2 = nn.Sequential(
            ConvUnit(3, 20, 7, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(20, 40, 5, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(40, 20, 5, bn=bn),
            ConvUnit(20, 10, 5, bn=bn),
        )

        self.column3 = nn.Sequential(
            ConvUnit(3, 24, 5, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(24, 48, 3, bn=bn),
            nn.MaxPool2d(2),
            ConvUnit(48, 24, 3, bn=bn),
            ConvUnit(24, 12, 3, bn=bn),
        )

        self.merge = nn.Sequential(
            ConvUnit(30, 1, 1, bn=bn)
        )

    def forward(self, x):
        x1 = self.column1(x)
        x2 = self.column2(x)
        x3 = self.column3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.merge(x)

        return x

