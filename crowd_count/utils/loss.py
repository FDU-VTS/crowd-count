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
        mae, mse = [0.0] * 2
        batch = len(output)
        for i in range(batch):
            sum_output = torch.sum(output[i])
            sum_gt = torch.sum(ground_truth[i])
            mae += abs(sum_output - sum_gt)
            mse += (sum_output - sum_gt) * (sum_output - sum_gt)
        return mae / batch, mse / batch


class EnlargeLoss:

    def __init__(self, number):
        super(EnlargeLoss).__init__()
        self.number = number

    def __call__(self, output, ground_truth):
        mae, mse = [0.0] * 2
        batch = len(output)
        for i in range(batch):
            sum_output = torch.sum(output[i] / self.number)
            sum_gt = torch.sum(ground_truth[i])
            mae += abs(sum_output - sum_gt)
            mse += mae ** 2
        return mae / batch, mse / batch
