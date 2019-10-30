from crowd_count.engine import train
from crowd_count.models import Res101
from crowd_count.data.data_loader import ShanghaiTechDataset
from crowd_count.utils import AVGLoss, TestLoss
import crowd_count.transforms as cc_transforms
import torchvision.transforms as transforms
import torch
device = "cuda: 0" if torch.cuda.is_available() else "cpu"


model = Res101()
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                         std=[0.23242045939, 0.224925786257, 0.221840232611])
                                    ])
gt_transform = cc_transforms.LabelEnlarge()
both_transform = cc_transforms.TransposeFlip()
train_set = ShanghaiTechDataset(mode="train", part="b", img_transform=img_transform, gt_transform=gt_transform, both_transform=both_transform)
test_set = ShanghaiTechDataset(mode="test", part="b", img_transform=img_transform)
train_loss = AVGLoss()
test_loss = TestLoss()
train(model, train_set, test_set, train_loss, test_loss, "Adam")
