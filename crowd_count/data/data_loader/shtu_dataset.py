from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm


class ShanghaiTechDataset(Dataset):

    def __init__(self,
                 mode="train",
                 part="a",
                 img_transform=None,
                 gt_transform=None,
                 both_transform=None,):
        self.root = {
            "a": {
                "train": "crowd_count/data/datasets/shtu_dataset/part_A_final/train_data/",
                "test": "crowd_count/data/datasets/shtu_dataset/part_A_final/test_data/",
            },
            "b": {
                "train": "crowd_count/data/datasets/shtu_dataset/part_B_final/train_data/",
                "test": "crowd_count/data/datasets/shtu_dataset/part_B_final/test_data/",
            }
        }[part][mode]
        self.paths = glob.glob(self.root + "images/*.jpg")
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
        pbar = tqdm(total=self.length)
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            img = Image.open(img_path).convert('RGB')
            with h5py.File(gt_path, 'r') as gt_file:
                den = np.asarray(gt_file['density'])
            result.append([img, den])
            pbar.update(1)
        pbar.close()
        return result
