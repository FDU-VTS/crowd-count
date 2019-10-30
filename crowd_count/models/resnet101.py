import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from crowd_count.models.network import ConvUnit

__all__ = ["Res101"]

class Res101(nn.Module):
    def __init__(self, pretrained=True):
        super(Res101, self).__init__()
        self.de_pred = nn.Sequential(
            ConvUnit(1024, 128, 1),
            nn.Conv2d(128, 1, kernel_size=1, stride=1)
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        res = models.resnet101(pretrained=pretrained)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.de_pred(x)
        x = F.upsample(x, scale_factor=8)
        return x


def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out