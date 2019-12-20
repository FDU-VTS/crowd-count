# -*- coding:utf-8 -*-
# ucf_cc_50 dataset

from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import h5py


class UCFCC50(Dataset):
    """Refer from `"UCF_50" <https://www.crcv.ucf.edu/data/ucf-cc-50/>`_ dataset

    """
    def __init__(self,
                 mode="train",
                 img_transform=None,
                 gt_transform=None,
                 both_transform=None,
                 dir="../crowd_count/data/datasets/ucf_cc_50/"):
        self.root = dir
        self.sum_path = glob.glob(self.root + "*.jpg")
        self.paths = self.sum_path[10:] if mode == 'train' else self.sum_path[:10]
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.both_transform = both_transform
        self.length = len(self.paths)
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
        den = den[np.newaxis, :]
        return img, den

    def load_data(self):
        result = []
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = Image.open(img_path).convert('RGB')
            with h5py.File(gt_path, 'r') as gt_file:
                den = np.asarray(gt_file['density'])
            result.append([img, den])

        return result
