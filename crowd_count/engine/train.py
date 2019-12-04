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
          train_batch=1,
          test_batch=1,
          num_worker=8,
          epoch_num=2000,
          learning_decay=0.995,
          saver=None,
          enlarge_num=1
          ):
    device = "cuda: {0}".format(cuda_num[0]) if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_flag is True:
        scheduler = StepLR(optimizer, step_size=1, gamma=learning_decay)
    if len(cuda_num) > 1:
        model = torch.nn.DataParallel(model, device_ids=cuda_num)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=num_worker)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=test_batch,
                             shuffle=False,
                             num_workers=num_worker)
    min_mae, min_mse = [1e10, 1e10]
    for epoch in range(epoch_num):
        sum_loss, temp_loss, sum_mae, sum_mse, sum_output, sum_gt = [0.0] * 6
        model = model.train()
        for i, (img, ground_truth) in enumerate(train_loader):
            img = img.float().to(device)
            ground_truth = ground_truth.float().to(device)
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
                print("| Train | lr: %.8f | output: %.1f | gt: %.1f |" % (optimizer.param_groups[0]['lr'],
                                                                  sum_output / (10 * train_batch * enlarge_num),
                                                                  sum_gt / (10 * train_batch * enlarge_num)))
                print("------------------------------------------------------")
                sum_output, sum_gt = [0.0] * 2
                temp_loss = sum_loss
        if scheduler_flag is True:
            scheduler.step()

        model = model.eval()
        with torch.no_grad():
            for img, ground_truth in iter(test_loader):
                img = img.float().to(device)
                ground_truth = ground_truth.float().to(device)
                output = model(img)
                mae, mse = test_loss(output, ground_truth)
                sum_mae += float(mae)
                sum_mse += float(mse)
            avg_mae = sum_mae / len(test_loader)
            avg_mse = math.sqrt(sum_mse / (len(test_loader)))
            if avg_mae < min_mae:
                min_mae = avg_mae
                min_mse = avg_mse
                if saver is not None:
                    saver.save(model, "mae_{mae}_mse_{mse}".format(mae=min_mae, mse=min_mse))
            print("********************** test ************************")
            print("* mae:%.1f, mse:%.1f, best_mae:%.1f, best_mse:%.1f *" % (avg_mae, avg_mse, min_mae, min_mse))
            print("* sum loss is {:.6f}                             *".format(sum_loss / len(test_loader)))
            print("****************************************************")
