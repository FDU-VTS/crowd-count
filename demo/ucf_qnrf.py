import sys

sys.path.append("../")
from crowdcount.engine import train
from crowdcount.models import *
from crowdcount.data.data_loader import *
from crowdcount.utils import *
import crowdcount.transforms as cc_transforms
import torchvision.transforms as transforms

model = Res101()
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.413525998592, 0.378520160913, 0.371616870165],
                                                         std=[0.284849464893, 0.277046442032, 0.281509846449])
                                    ])
gt_transform = cc_transforms.LabelEnlarge(100)
both_transform = cc_transforms.ComplexCompose([
    cc_transforms.AutoRotation(),
    cc_transforms.Scale((688, 1024)),
    cc_transforms.TransposeFlip()
])
train_set = UCFQNRF(mode="train",
                    img_transform=img_transform,
                    gt_transform=gt_transform,
                    both_transform=both_transform,
                    root="../crowdcount/data/datasets/UCF-QNRF_ECCV18/")
test_set = UCFQNRF(mode="test",
                   img_transform=img_transform,
                   gt_transform=gt_transform,
                   both_transform=both_transform,
                   root="../crowdcount/data/datasets/UCF-QNRF_ECCV18/")
train_loss = loss.Compose([[1, AVGLoss()], [1, RankLoss(100)]])
test_loss = EnlargeLoss(100)
saver = Saver(mode="remain", path="../exp/03-02-ucf_qnrf-batch2-rank-2")
tb = TensorBoard(path="../runs/03-02-ucf_qnrf-batch2-rank-2")
train(model, train_set, test_set, train_loss, test_loss, optim="Adam", saver=saver, cuda_num=[3],
      train_batch=2, test_batch=1, learning_rate=1e-5, epoch_num=250, enlarge_num=100, scheduler_flag=True)
