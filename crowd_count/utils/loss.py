# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch.nn as nn
import torch


class AVGLoss:

    def __init__(self):
        super(AVGLoss).__init__()

    def __call__(self, output, ground_truth):
        batch = len(output)
        loss_function = nn.MSELoss(reduction='mean')
        loss = loss_function(output.squeeze(), ground_truth.squeeze())
        return loss / (2 * batch)


class SUMLoss:

    def __init__(self):
        super(SUMLoss).__init__()

    def __call__(self, output, ground_truth):
        batch = len(output)
        loss_function = nn.MSELoss(reduction='sum')
        loss = loss_function(output.squeeze(), ground_truth.squeeze())
        return loss / (2 * batch)


class TestLoss:

    def __init__(self):
        super(TestLoss).__init__()

    def __call__(self, output, ground_truth):
        sum_output = torch.sum(output)
        sum_gt = torch.sum(ground_truth)
        mae = abs(sum_output - sum_gt)
        mse = (sum_output - sum_gt) * (sum_output - sum_gt)
        return mae, mse


class EnlargeLoss:

    def __init__(self):
        super(EnlargeLoss).__init__()

    def __call__(self, output, ground_truth, number):
        sum_output = torch.sum(output / number)
        sum_gt = torch.sum(ground_truth)
        mae = abs(sum_output - sum_gt)
        mse = mae ** 2
        return mae, mse
