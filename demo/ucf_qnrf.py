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
                                    transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                         std=[0.23242045939, 0.224925786257, 0.221840232611])
                                    ])
gt_transform = cc_transforms.LabelEnlarge()
both_transform = cc_transforms.ComplexCompose([cc_transforms.TransposeFlip(),
                                               cc_transforms.Scale((512, 512))])
test_transform = cc_transforms.Scale((512, 512))
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
train_loss = AVGLoss()
test_loss = EnlargeLoss(100)
saver = Saver(mode="remain", path="../exp/11-27-ucf_qnrf")
train(model, train_set, test_set, train_loss, test_loss, optim="Adam", saver=saver, cuda_num=[0, 1, 2, 3],
      train_batch=4,test_batch=4, learning_rate=1e-5, enlarge_num=100, scheduler_flag=True)
