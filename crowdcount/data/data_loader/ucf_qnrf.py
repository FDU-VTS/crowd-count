# -*- coding:utf-8 -*-
# ucf_qnrf dataset

from .crowd_dataset import CrowdDataset
import glob
import numpy as np
import h5py
import skimage.io
import skimage.color
import skimage.transform
from tqdm import tqdm
import os


class UCFQNRF(CrowdDataset):

    def __init__(self, mode="train", img_transform=None, gt_transform=None, both_transform=None,
                 root="./crowd_count/data/datasets/UCF-QNRF_ECCV18/"):
        super().__init__(img_transform,  gt_transform, both_transform)
        self.mode = mode
        self.root = os.path.join(root, "Train/") if mode == "train" else os.path.join(root, "Test/")
        self.paths = glob.glob(self.root + "*.jpg")
        self.load_data()

    def load_data(self):
        print("******************ucf_qnrf loading******************")
        pbar = tqdm(total=len(self.paths))
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                continue
            img = skimage.io.imread(img_path, plugin='matplotlib')
            img = skimage.color.grey2rgb(img)
            with h5py.File(gt_path, 'r') as gt_file:
                den = np.asarray(gt_file['density'])
            self.dataset.append([img, den])
            pbar.update(1)
        pbar.close()
