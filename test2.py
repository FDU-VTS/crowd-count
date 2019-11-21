import sys
sys.path.append("../")
from crowd_count.engine import train
from crowd_count.models import *
from crowd_count.data.data_loader import *
from crowd_count.utils import *
import crowd_count.transforms as cc_transforms
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim
import torch.nn
from torch.optim.lr_scheduler import StepLR
import math
device="cuda:0"

model_path = "/home/vts/chensongjian/CrowdCount/exp/best_models/shtu_a_mae_54.2_mse_69.1.pt"
model = Res101().to('cuda:0')
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                         std=[0.23242045939, 0.224925786257, 0.221840232611])
                                    ])
gt_transform = cc_transforms.LabelEnlarge()
test_transform = cc_transforms.Scale((512, 512))
test_set = ShanghaiTechDataset(mode="test", part="a", img_transform=img_transform, both_transform=test_transform, dir="./crowd_count/data/datasets/shtu_dataset/")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.load_state_dict(torch.load(model_path))
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=8)
mae = 0
with torch.no_grad():
    for img, ground_truth in iter(test_loader):
        img = img.float().to(device)
        ground_truth = ground_truth.float().to(device)
        output = model(img)
        mae += abs(torch.sum(output / 100) - torch.sum(ground_truth))
    print(mae / 182)
