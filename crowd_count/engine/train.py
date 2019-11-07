# -*- coding: utf-8 -*-
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
          cuda_num=[0],
          optim: str = "Adam",
          scheduler_flag=True,
          learning_rate=1e-5,
          weight_decay=1e-4,
          batch_size=1,
          num_worker=8,
          epoch_num=2000,
          learning_decay=0.995,
          saver=None,
          ):
    model = model.cuda()
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_flag is True:
        scheduler = StepLR(optimizer, step_size=1, gamma=learning_decay)
    if len(cuda_num) > 1:
        model = torch.nn.DataParallel(model, device_ids=cuda_num)
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
        sum_loss, temp_loss, sum_mae, sum_mse, sum_output, sum_gt = [0.0] * 6
        model = model.train()
        for i, (img, ground_truth) in enumerate(train_loader):
            img = img.float().cuda()
            ground_truth = ground_truth.float().cuda()
            optimizer.zero_grad()
            output = model(img)
            sum_output += torch.sum(output)
            sum_gt += torch.sum(ground_truth)
            loss = train_loss(output, ground_truth)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
            if i % 10 == 9 or i == len(train_loader) - 1:
                print("| epoch: {} / {} | batch: {} / {} | loss: {:.6f} |".format(
                    epoch, epoch_num, i + 1, len(train_loader), (sum_loss - temp_loss) / 10))
                print("| output: %.1f | gt: %.1f |" % (sum_output / (1000 * batch_size), sum_gt / (1000 * batch_size)))
                print("------------------------------------------------------")
                sum_output, sum_gt = [0.0] * 2
                temp_loss = sum_loss
        if scheduler_flag is True:
            scheduler.step()

        model = model.eval()
        for img, ground_truth in iter(test_loader):
            img = img.float().cuda()
            ground_truth = ground_truth.float().cuda()
            output = model(img)
            mae, mse = test_loss(output, ground_truth, 100)
            sum_mae += float(mae)
            sum_mse += float(mse)
        if sum_mae / (len(test_loader) * batch_size) < min_mae:
            min_mae = sum_mae / (len(test_loader) * batch_size)
            min_mse = math.sqrt(sum_mse / (len(test_loader) * batch_size))
            if saver is not None:
                saver.save(model, "mae_{mae}_mse_{mse}".format(mae=min_mae, mse=min_mse))
        print("********************** test ************************")
        print("* mae:%.1f, mse:%.1f, best_mae:%.1f, best_mse:%.1f *" %
              (sum_mae / (len(test_loader) * batch_size), math.sqrt(sum_mse / (len(test_loader) * batch_size)),
               min_mae, min_mse))
        print("* sum loss is {:.6f}                             *".format(sum_loss / len(test_loader)))
        print("****************************************************")
