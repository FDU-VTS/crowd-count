from crowdcount.data.data_loader import *
import crowdcount.transforms as cc_transforms
import torchvision.transforms as transforms


def build_data(cfg):
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.452016860247, 0.447249650955, 0.431981861591],
                                                             std=[0.23242045939, 0.224925786257, 0.221840232611])
                                        ])
    gt_transform = cc_transforms.LabelEnlarge(100)
    both_transform = cc_transforms.TransposeFlip()

    return train_set, test_set
