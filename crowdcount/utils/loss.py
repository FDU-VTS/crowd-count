# -*- coding: utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
import torch.nn as nn
import torch


class AVGLoss:
    """ The loss function used in train part, torch.nn.MSELoss with reduction='mean', which means the sum of the density map will be divided by
    the number of pixels in the density map. It can be described as:

    .. math::
        \ell(\\theta) = \\frac{1}{2N*B}\sum_{i=1}^N\parallel{Z(X_i;\\theta) - Z_i^GT}\parallel_2^2

    """

    def __init__(self):
        super(AVGLoss).__init__()

    def __call__(self, output, ground_truth):
        """
        Args:
            output (torch.tensor): the output generated by model
            ground_truth (torch.tensor): the ground truth compared with output

        Return:
            float
        """
        batch = len(output)
        loss_function = nn.MSELoss(reduction='mean')
        loss = loss_function(output.squeeze(), ground_truth.squeeze())
        return loss / (2 * batch)


class SUMLoss:
    """ The loss function used in train part, torch.nn.MSELoss with reduction='sum', which means the density map will be summed.
    It can be described as:

    .. math::
        \ell(\\theta) = \\frac{1}{2*B}\sum_{i=1}^N\parallel{Z(X_i;\\theta) - Z_i^GT}\parallel_2^2

    """

    def __init__(self):
        super(SUMLoss).__init__()

    def __call__(self, output, ground_truth):
        """
        Args:
            output (torch.tensor): the output generated by model
            ground_truth (torch.tensor): the ground truth compared with output

        Return:
            float
        """
        batch = len(output)
        loss_function = nn.MSELoss(reduction='sum')
        loss = loss_function(output.squeeze(), ground_truth.squeeze())
        return loss / (2 * batch)


class TestLoss:
    """The loss function used in test part, this loss just get mae and mse with (output-ground_truth) and mae ** 2

    """

    def __init__(self):
        super(TestLoss).__init__()

    def __call__(self, output, ground_truth):
        """
        Args:
            output (torch.tensor): the output generated by model
            ground_truth (torch.tensor): the ground truth compared with output

        Return:
            float
        """
        mae, mse = [0.0] * 2
        batch = len(output)
        for i in range(batch):
            sum_output = torch.sum(output[i])
            sum_gt = torch.sum(ground_truth[i])
            mae += abs(sum_output - sum_gt)
            mse += (sum_output - sum_gt) * (sum_output - sum_gt)
        return mae / batch, mse / batch


class EnlargeLoss:
    """When you enlarge the density map with scale factor (10, 100 or 1000), get test loss with this function.

    Args:
        number (int): the scale factor used to enlarge the density map

    """

    def __init__(self, number):
        super(EnlargeLoss).__init__()
        self.number = number

    def __call__(self, output, ground_truth):
        """
        Args:
            output (torch.tensor): the output generated by model, will be divided by self.number
            ground_truth (torch.tensor): the ground truth compared with output

        Return:
            float
        """
        mae, mse = [0.0] * 2
        batch = len(output)
        for i in range(batch):
            sum_output = torch.sum(output[i] / self.number)
            sum_gt = torch.sum(ground_truth[i])
            mae += abs(sum_output - sum_gt)
            mse += mae ** 2
        return mae / batch, mse / batch
