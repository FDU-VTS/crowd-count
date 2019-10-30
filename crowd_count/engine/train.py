# -*- coding: utf-8 -*-
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim
import torch.nn
from torch.optim.lr_scheduler import StepLR
import math


def train(model,
          train_set: Dataset,
          test_set: Dataset,
          train_loss,
          test_loss,
          optim: str = None,
          scheduler_flag=False,
          learning_rate=1e-5,
          weight_decay=1e-4,
          batch_size=1,
          shuffle=True,
          num_worker=8,
          epoch_num=2000,
          learning_decay=0.995,
          ):
    model = model.cuda()
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_flag is True:
        scheduler = StepLR(optimizer, step_size=1, gamma=learning_decay)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_worker)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_worker)
    min_mae, min_mse = [1e10, 1e10]
    for epoch in range(epoch_num):
        sum_loss, temp_loss, sum_mae, sum_mse = [0.0] * 4
        model = model.train()
        for i, (img, ground_truth) in enumerate(train_loader):
            img = img.float().cuda()
            ground_truth = ground_truth.float().cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = train_loss(output.squeeze(), ground_truth.squeeze())
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
            if i % 10 == 9 or i == len(train_loader) - 1:
                print("{0} / {1} have done, loss is {2}".format(i + 1, len(train_loader), (sum_loss - temp_loss) / 10))
                temp_loss = sum_loss
        if scheduler_flag is True:
            scheduler.step()

        for input, ground_truth in iter(test_loader):
            input = input.float().cuda()
            ground_truth = ground_truth.float().cuda()
            output = model(input)
            mae, mse = test_loss(output, ground_truth)
            sum_mae += float(mae)
            sum_mse += float(mse)
        if sum_mae / len(test_loader) < min_mae:
            min_mae = sum_mae / len(test_loader)
            min_mse = math.sqrt(sum_mse / len(test_loader))
        print("mae:%.1f, mse:%.1f, best_mae:%.1f, best_mse:%.1f" %
              (sum_mae / len(test_loader), math.sqrt(sum_mse / len(test_loader)), min_mae, min_mse))
        print("sum loss is {0}".format(sum_loss / len(test_loader)))
