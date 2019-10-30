# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch.nn as nn
import torch


class AVGLoss():

    def __init__(self):
        super(AVGLoss).__init__()

    def __call__(self, output, ground_truth):
        number = len(output)
        loss_function = nn.MSELoss(reduction='mean')
        loss = loss_function(output, ground_truth)
        return loss / (2 * number)


class SUMLoss():

    def __init__(self):
        super(SUMLoss).__init__()

    def __call__(self, output, ground_truth):
        number = len(output)
        loss_function = nn.MSELoss(reduction='sum')
        loss = loss_function(output, ground_truth)
        return loss / (2 * number)


class TestLoss():

    def __init__(self):
        super(TestLoss).__init__()

    def __call__(self, output, ground_truth):
        sum_output = torch.sum(output)
        sum_gt = torch.sum(ground_truth)
        mae = abs(sum_output - sum_gt)
        mse = (sum_output - sum_gt) * (sum_output - sum_gt)
        return mae, mse


class EnlargeLoss(nn.Module):

    def __init__(self):
        super(EnlargeLoss).__init__()

    def forward(self, output, ground_truth, number):
        sum_output = torch.sum(output / number)
        sum_gt = torch.sum(ground_truth)
        mae = abs(sum_output - sum_gt)
        mse = (sum_output - sum_gt) * (sum_output - sum_gt)
        return mae, mse

