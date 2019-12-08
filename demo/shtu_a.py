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
gt_transform = cc_transforms.SingleCompose([cc_transforms.LabelEnlarge()])
both_transform = cc_transforms.ComplexCompose([cc_transforms.TransposeFlip(),
                                               cc_transforms.Scale([512, 512])])
test_transform = None
train_set = ShanghaiTechDataset(mode="train", part="a", img_transform=img_transform, gt_transform=gt_transform, both_transform=both_transform)
test_set = ShanghaiTechDataset(mode="test", part="a", img_transform=img_transform, both_transform=test_transform)
train_loss = AVGLoss()
test_loss = EnlargeLoss(100)
saver = Saver(mode="remain", path="../exp/11-27-shtu_a")
train(model, train_set, test_set, train_loss, test_loss, optim="Adam", saver=saver, cuda_num=[3], train_batch=1,
      test_batch=1, learning_rate=1e-5, enlarge_num=100, scheduler_flag=True)
