import sys

sys.path.append("../")
from crowdcount.engine import *
from crowdcount.models import *
from crowdcount.data.data_loader import *
from crowdcount.utils import *
import crowdcount.transforms as cc_transforms
import torchvision.transforms as transforms

model = Res101()
model_path = "/home/vts/chensongjian/CrowdCount/exp/2019-2-17-shtu_b/mae_8.840841251083567_mse_15.259539232482005.pt"
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.413525998592, 0.378520160913, 0.371616870165],
                                                         std=[0.284849464893, 0.277046442032, 0.281509846449])
                                    ])
gt_transform = cc_transforms.LabelEnlarge(100)

test_set = UCFQNRF(mode="test",
                   img_transform=img_transform,
                   gt_transform=gt_transform,
                   root="../crowdcount/data/datasets/UCF-QNRF_ECCV18/")
# test_set = ShanghaiTechMatlab(mode="test",
#                               img_transform=img_transform,
#                               gt_transform=gt_transform,
#                               data_path="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/shanghaitech_part_B")
test_loss = EnlargeLoss(100)
evaluate(model, model_path, test_set, test_loss, cuda_num=[0], test_batch=1)
