from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import h5py


class SHTUMask(Dataset):

    def __init__(self, mode="train", part="a", **kwargs):
        self.root = {
            "a": {
                "train": "../data/shtu_dataset/part_A_final/train_data/",
                "test": "../data/shtu_dataset/part_A_final/test_data/",
            },
            "b": {
                "train": "../data/shtu_dataset/part_B_final/train_data/",
                "test": "../data/shtu_dataset/part_B_final/test_data/",
            }
        }[part][mode]
        self.paths = glob.glob(self.root + "images/*.jpg")
        self.zoom_size = kwargs['zoom_size']
        self.transform = kwargs['transform']
        self.dataset = self.load_data()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img, den = self.dataset[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, den

    def load_data(self):
        result = []
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            mask_path = gt_path.replace('ground_truth', 'mask')
            img = Image.open(img_path).convert('RGB')
            with h5py.File(gt_path, 'r')  as gt_file:
                den = np.asarray(gt_file['density'])
            with h5py.File(mask_path) as mask_file:
                mask = mask_file['mask']
            img[mask != 0] == 255
            den = den[np.newaxis, :]
            result.append([img, den])
        return result
