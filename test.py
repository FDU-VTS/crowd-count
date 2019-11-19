from crowd_count.models import Res101
from crowd_count.data.data_loader import ShanghaiTechDataset, SHTUMask, ShanghaiTechDatasetA
from crowd_count.utils import AVGLoss, EnlargeLoss, Saver
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn
import math
import matplotlib.pyplot as plt
import numpy as np
# 7.8 16.4

mask_path = "/home/vts/chensongjian/CrowdCount/exp/best_models/shtu_mask_b_resnet_101_best_model.pt"
best_path = "/home/vts/chensongjian/CrowdCount/exp/best_models/shtu_b_resnet101_best_model.pt"
mask_model = Res101().to("cuda:1")
mask_model = torch.nn.DataParallel(mask_model, [1, 2, 3])
mask_model.load_state_dict(torch.load(mask_path))
mask_model = mask_model.eval()
best_model = Res101().to("cuda:1")
best_model = torch.nn.DataParallel(best_model, [1, 2, 3])
best_model.load_state_dict(torch.load(best_path))
best_model = best_model.eval()
test_loss = EnlargeLoss(100)
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                         std=[0.23242045939, 0.224925786257, 0.221840232611])
                                    ])
test_set = ShanghaiTechDataset(mode="test", part="b", img_transform=img_transform)
mask_set = SHTUMask(mode='test', part='b', img_transform=img_transform)
test_loader = DataLoader(dataset=test_set,
                         batch_size=2,
                         shuffle=False,
                         num_workers=8)
mask_loader = DataLoader(dataset=mask_set,
                         batch_size=2,
                         shuffle=False,
                         num_workers=8)
sum_mae, sum_mse = [0.0] * 2
normal_mae =[]
mask_mae = []
result = []
for img, ground_truth in iter(test_loader):
    img = img.float().to("cuda:2")
    ground_truth = ground_truth.float().to("cuda:2")
    output = best_model(img)
    mae, mse = test_loss(output, ground_truth)
    normal_mae.append([float(mae), float(mse)])
    sum_mae += float(mae)
    sum_mse += float(mse)
print(sum_mae / (len(test_loader) * 2))
print(math.sqrt(sum_mse / (len(test_loader) * 2)))
sum_mae, sum_mse = [0.0] * 2
for img, ground_truth in iter(mask_loader):
    img = img.float().to("cuda:2")
    ground_truth = ground_truth.float().to("cuda:2")
    output = best_model(img)
    mae, mse = test_loss(output, ground_truth)
    mask_mae.append([float(mae), float(mse)])
    sum_mae += float(mae)
    sum_mse += float(mse)
print(sum_mae / (len(test_loader) * 2))
print(math.sqrt(sum_mse / (len(test_loader) * 2)))
for i in range(len(normal_mae)):
    n_mae, n_mse = normal_mae[i]
    m_mae, m_mse = mask_mae[i]
    if n_mae < m_mae:
        result.append([n_mae, n_mse])
    else:
        result.append([m_mae, m_mse])
result = np.asarray(result)
print(np.sum(result, axis=0) / (len(test_loader) * 2))

