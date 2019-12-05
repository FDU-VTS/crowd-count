from torch.utils.data import Dataset
import os
import PIL.Image as Image
import h5py
import numpy as np


class FudanDataset(Dataset):

    def __init__(self,
                 mode="train",
                 img_transform=None,
                 gt_transform=None,
                 both_transform=None,
                 dir="../crowd_count/data/datasets/FDST_dataset/"):
        self.root = os.path.join(dir, "train_data/") if mode == "train" else os.path.join(dir, "test_data/")
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
