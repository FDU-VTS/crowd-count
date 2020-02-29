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
gt_transform = cc_transforms.LabelEnlarge(100)
both_transform = cc_transforms.TransposeFlip()
# train_set = ShanghaiTechDataset(mode="train",
#                                 part="b",
#                                 img_transform=img_transform,
#                                 gt_transform=gt_transform,
#                                 both_transform=both_transform,
#                                 root="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/shtu_dataset_sigma_15")
# test_set = ShanghaiTechDataset(mode="test",
#                                part='b',
#                                img_transform=img_transform,
#                                gt_transform=gt_transform,
#                                root="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/shtu_dataset_sigma_15")
train_set = ShanghaiTechMatlab(mode="train",
                               img_transform=img_transform,
                               gt_transform=gt_transform,
                               main_transform=both_transform,
                               data_path="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/shanghaitech_part_B")
test_set = ShanghaiTechMatlab(mode="test",
                              img_transform=img_transform,
                              gt_transform=gt_transform,
                              data_path="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/shanghaitech_part_B")
train_loss = loss.Compose([AVGLoss(), RankLoss(100)])
test_loss = EnlargeLoss(100)
saver = Saver(path="../exp/2019-2-29-shtu_b-rank")
tb = TensorBoard(path="../runs/2019-2-29-shtu_b-rank")
train(model, train_set, test_set, train_loss, test_loss, optim="Adam", saver=saver, cuda_num=[0, 1], train_batch=4,
      test_batch=4, learning_rate=1e-5, epoch_num=1000, enlarge_num=100, tensorboard=tb)
