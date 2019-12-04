import numpy as np


class CrowdDataset(object):

    def __init__(self,
                 img_transform=None,
                 gt_transform=None,
                 both_transform=None):
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.both_transform = both_transform
        self.dataset = []

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
        raise NotImplementedError
