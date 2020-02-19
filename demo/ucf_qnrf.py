import sys

sys.path.append("../")
from crowdcount.engine import train
from crowdcount.models import *
from crowdcount.data.data_loader import *
from crowdcount.utils import *
import crowdcount.transforms as cc_transforms
import torchvision.transforms as transforms

model = CSRNet()
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.413525998592, 0.378520160913, 0.371616870165],
                                                         std=[0.284849464893, 0.277046442032, 0.281509846449])
                                    ])
gt_transform = cc_transforms.LabelEnlarge(100)
both_transform = cc_transforms.TransposeFlip()
# train_set = UCFQNRF(mode="train",
#                     img_transform=img_transform,
#                     gt_transform=gt_transform,
#                     both_transform=both_transform,
#                     root="../crowdcount/data/datasets/UCF-QNRF_ECCV18/")
# test_set = UCFQNRF(mode="test",
#                    img_transform=img_transform,
#                    gt_transform=gt_transform,
#                    both_transform=both_transform,
#                    root="../crowdcount/data/datasets/UCF-QNRF_ECCV18/")
train_set = ShanghaiTechMatlab(mode="train",
                               img_transform=img_transform,
                               gt_transform=gt_transform,
                               main_transform=both_transform,
                               data_path="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/UCF-qnrf")
test_set = ShanghaiTechMatlab(mode="test",
                              img_transform=img_transform,
                              gt_transform=gt_transform,
                              data_path="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/UCF-qnrf")
train_loss = AVGLoss()
test_loss = EnlargeLoss(100)
saver = Saver(mode="remain", path="../exp/02-17-ucf_qnrf-batch1")
tb = TensorBoard(path="../runs/02-17-ucf_qnrf-batch1")
train(model, train_set, test_set, train_loss, test_loss, optim="Adam", saver=saver, cuda_num=[1],
      train_batch=1,test_batch=1, learning_rate=1e-5, enlarge_num=100, scheduler_flag=True)
