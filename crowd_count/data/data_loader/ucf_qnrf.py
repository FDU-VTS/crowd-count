# -*- coding:utf-8 -*-
# ucf_qnrf dataset

from torch.utils.data import Dataset
import glob
import numpy as np
import h5py
import skimage.io
import skimage.color
import skimage.transform
from tqdm import tqdm


class UCFQNRF(Dataset):

    def __init__(self, mode="train", img_transform=None, gt_transform=None, both_transform=None):
        self.mode = mode
        self.root = "./crowd_count/data/datasets/UCF-QNRF_ECCV18/Train/" if mode == "train" else \
                "./crowd_count/data/datasets/UCF-QNRF_ECCV18/Test/"
        self.paths = glob.glob(self.root + "*.jpg")
        self.length = len(self.paths)
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.both_transform = both_transform
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
        pbar = tqdm(total=self.length)
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = skimage.io.imread(img_path, plugin='matplotlib')
            img = skimage.color.grey2rgb(img)
            with h5py.File(gt_path, 'r') as gt_file:
                try:
                    den = np.asarray(gt_file['density'])
                except:
                    print(img.shape, img_path)
                    continue
            if self.mode == "test":
                # den = np.asarray(den)
                # img = np.asarray(img)
                # height = den.shape[0]
                # width = den.shape[1]
                # h = height // 2
                # w = width // 2
                # result.append([img[: h, : w], den[: h, : w]])
                # result.append([img[: h, w:], den[: h, w:]])
                # result.append([img[h:, : w], den[h:, : w]])
                # result.append([img[h:, w:], den[h:, w:]])
                result.append([img, den])
            elif self.mode == "train":
                if den.shape[0] < 512 or den.shape[1] < 512:
                    continue
                else:
                    result.append([img, den])
            pbar.update(1)
        self.length = len(result)
        pbar.close()
        return result
