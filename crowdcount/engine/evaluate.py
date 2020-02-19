# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim
import torch.nn
import math


def evaluate(model,
             model_path,
             test_set: Dataset,
             test_loss,
             cuda_num=[0],
             test_batch=1,
             num_worker=8,
             ):
    """start to eval
    Args:
        model (torch.nn.Module): the model built to train.
        test_set (torch.utils.data.Dataset or object): test dataset constructed into torch.utils.data.DataLoader.
        test_set (torch.utils.data.Dataset or object): test dataset constructed into torch.utils.data.DataLoader.
        cuda_num (list, optional): CUDA devices(default: [0]).
        test_batch (int, optional): test batch(default: 1).
        num_worker (int, optional): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process(default: 8).
    """
    if cuda_num is None:
        cuda_num = [0]
    device = "cuda: {0}".format(cuda_num[0]) if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    test_loader = DataLoader(dataset=test_set,
                             batch_size=test_batch,
                             shuffle=False,
                             num_workers=num_worker)
    sum_mae, sum_mse = 0.0, 0.0
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
        print("********************** test ************************")
        print("* mae:%.1f, mse:%.1f, *" % (avg_mae, avg_mse))
        print("****************************************************")
