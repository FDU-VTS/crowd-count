from torch.utils.data import Dataset
import os
from PIL import Image
import h5py
import numpy as np


class FudanDataset(Dataset):
    """Fudan-ShanghaiTech Dataset,
    Refer from `"Locality-constrained..." <https://arxiv.org/pdf/1907.07911.pdf>` _paper.
    include 100 videos captured from 13 different scenes, and FDST dataset contains 150,000 frames,
    with a total of 394,081 annotated heads

    Args:
        mode (str, optional): "train" | "test", if "train": load the train part data,
            if "test": load the test part data(default: "train").
        img_transform (list of crowdcount.transform objects, optional): transforms applied to image(default: None).
        gt_transform (list of crowdcount.transform objects, optional): transforms applied to ground truth(default: None).
        both_transform (list of crowdcount.transform objects, optional):
            transforms applied to both of image and ground truth(default:None).
        root (str, optional): the root directory of dataset(default: "../crowd_count/data/datasets/shtu_dataset/").
    """

    def __init__(self,
                 mode="train",
                 img_transform=None,
                 gt_transform=None,
                 both_transform=None,
                 root="../crowd_count/data/datasets/FDST_dataset/"):
        self.root = os.path.join(root, "train_data/") if mode == "train" else os.path.join(root, "test_data/")
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.both_transform = both_transform
        self.length = 0
        self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img, den = self.dataset[item]
        if self.both_transform is not None:
            img, den = self.both_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def load_data(self):
        result = []
        print("******************fudan_data loading******************")
        for root, dirs, files in os.walk(self.root, topdown=False):
            for file in files:
                img_path = os.path.join(root, file)
                gt_path = img_path.replace('.jpg', '.h5')
                if not os.path.exists(img_path) or not os.path.exists(gt_path):
                    raise IOError("{} does not exist".format(img_path))
                img = Image.open(img_path).convert('RGB')
                with h5py.File(gt_path) as gt_file:
                    den = np.asarray(gt_file['density'])
                result.append([img, den])
                self.length += 1
        return result
