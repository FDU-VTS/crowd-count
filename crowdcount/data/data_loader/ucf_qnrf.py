# -*- coding:utf-8 -*-
# ucf_qnrf dataset

import glob
import numpy as np
import h5py
import skimage.io
import skimage.color
from tqdm import tqdm
import os
from torch.utils.data import Dataset


class UCFQNRF(Dataset):

    def __init__(self, mode="train", img_transform=None, gt_transform=None, both_transform=None,
                 root="./crowd_count/data/datasets/UCF-QNRF_ECCV18/"):
        self.mode = mode
        self.root = os.path.join(root, "Train/") if mode == "train" else os.path.join(root, "Test/")
        self.paths = glob.glob(self.root + "*.jpg")
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.both_transform = both_transform
        self.dataset = []
        self.load_data()

    def __len__(self):
        return len(self.dataset)

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
        print("******************ucf_qnrf loading******************")
        pbar = tqdm(total=len(self.paths))
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = skimage.io.imread(img_path, plugin='matplotlib')
            img = skimage.color.grey2rgb(img)
            with h5py.File(gt_path, 'r') as gt_file:
                den = np.asarray(gt_file['density'])
            self.dataset.append([img, den])
            pbar.update(1)
        pbar.close()
