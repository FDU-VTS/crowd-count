# -*- coding:utf-8 -*-
# ------------------------
# written by Songjian Chen
# 2018-10
# ------------------------
from crowd_count.data.data_loader import ShanghaiTechDataset
import crowd_count.utils as utils
import crowd_count.transforms as cc_transforms
from crowd_count.models import Res101
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn
from torch.optim.lr_scheduler import StepLR
import math
from torchvision import transforms
import yaml

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_param():
    param = yaml.load(open("evaluation/resnet101/parameters.yaml", 'r'), Loader=yaml.FullLoader)
    cuda_num = list(param["cuda"])
    train_batch = int(param["train_batch"])
    test_batch = int(param["test_batch"])
    epoch_num = int(param["epoch_num"])
    learning_rate = float(param["learning_rate"])
    learning_decay = float(param["learning_decay"])
    return cuda_num, train_batch, test_batch, epoch_num, learning_rate, learning_decay


def _load_data(train_batch, test_batch):
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                             std=[0.23242045939, 0.224925786257, 0.221840232611])
                                        ])
    # gt_transform = cc_transforms.ResizeShrink()
    gt_transform = cc_transforms.LabelEnlarge()
    both_transform = cc_transforms.TransposeFlip()
    print("dataset loading .........")
    train_dataset = ShanghaiTechDataset(mode="train", part='b', img_transform=img_transform, gt_transform=gt_transform, both_transform=both_transform)
    test_dataset = ShanghaiTechDataset(mode="test", part='b', img_transform=img_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=8)
    return train_loader, test_loader


def train():
    # load paramters
    cuda_num, train_batch, test_batch, epoch_num, learning_rate, learning_decay = _load_param()
    train_loader, test_loader = _load_data(train_batch, test_batch)
    net = Res101().cuda()
    if len(cuda_num) > 1:
        net = torch.nn.DataParallel(net, device_ids=cuda_num)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=learning_decay)
    print("start to train net.....")
    min_mae, min_mse = [1e10, 1e10]
    for epoch in range(epoch_num):
        sum_loss, temp_loss, sum_mae, sum_mse = [0.0] * 4
        print("{0} epoches / {1} epoches are done".format(epoch, epoch_num))
        net = net.train()
        for i, (input, ground_truth) in enumerate(train_loader):
            input = input.float().cuda()
            ground_truth = ground_truth.float().cuda()
            optimizer.zero_grad()
            output = net(input)
            loss = utils.avg_mse(output.squeeze(), ground_truth.squeeze())
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
            if i % 10 == 9 or i == len(train_loader) - 1:
                print("{0} / {1} have done, loss is {2}".format(i + 1, len(train_loader), (sum_loss - temp_loss) / 10))
                temp_loss = sum_loss
        scheduler.step()

        # test model
        net = net.eval()
        for input, ground_truth in iter(test_loader):
            input = input.float().cuda()
            ground_truth = ground_truth.float().cuda()
            output = net(input)
            mae, mse = utils.test_loss(output, ground_truth, 100)
            sum_mae += float(mae)
            sum_mse += float(mse)
        if sum_mae / len(test_loader) < min_mae:
            min_mae = sum_mae / len(test_loader)
            min_mse = math.sqrt(sum_mse / len(test_loader))
        print("mae:%.1f, mse:%.1f, best_mae:%.1f, best_mse:%.1f" %
              (sum_mae / len(test_loader), math.sqrt(sum_mse / len(test_loader)), min_mae, min_mse))
        print("sum loss is {0}".format(sum_loss / len(test_loader)))
