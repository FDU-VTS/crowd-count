import sys

sys.path.append("../")
from crowdcount.engine import *
from crowdcount.models import *
from crowdcount.data.data_loader import *
from crowdcount.utils import *
import crowdcount.transforms as cc_transforms
import torchvision.transforms as transforms

model = Res101()
model_path = "/home/vts/chensongjian/CrowdCount/exp/02-20-ucf_qnrf-batch4/mae_110.08521439786443_mse_192.1708243146396.pt"
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                         std=[0.23242045939, 0.224925786257, 0.221840232611])
                                    ])
gt_transform = cc_transforms.LabelEnlarge(100)

test_set = ShanghaiTechMatlab(mode="test",
                              img_transform=img_transform,
                              gt_transform=gt_transform,
                              data_path="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/shanghaitech_part_B")
test_loss = EnlargeLoss(100)
evaluate(model, model_path, test_set, test_loss, cuda_num=[2, 3], mode="multi", test_batch=1)
