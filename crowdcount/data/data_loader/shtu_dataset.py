import glob
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import os
from .crowd_dataset import CrowdDataset


class ShanghaiTechDataset(CrowdDataset):
    """ShanghaiTech Dataset,
    Refer from `"MCNN..." <https://www.semanticscholar.org/paper/Single-Image-Crowd-Counting-via-Multi-Column-Neural-Zhang-Zhou/2dc3b3eff8ded8914c8b536d05ee713ff0cdf3cd>`_ paper.

    Args:
        mode (str, optional): "train" | "test", if "train": load the train part data,
            if "test": load the test part data(default: "train").
        part (str, optional): "a" | "b", if "a": load the ShanghaiTech part A,
            if "b": load the shanghaiTech part B(default: "a").
        img_transform (list of crowdcount.transform objects, optional): transforms applied to image(default: None).
        gt_transform (list of crowdcount.transform objects, optional): transforms applied to ground truth(default: None).
        both_transform (list of crowdcount.transform objects, optional):
            transforms applied to both of image and ground truth(default:None).
        root (str, optional): the root directory of dataset(default: "../crowd_count/data/datasets/shtu_dataset/").
    """

    def __init__(self,
                 mode="train",
                 part="a",
                 img_transform=None,
                 gt_transform=None,
                 both_transform=None,
                 root="../crowd_count/data/datasets/shtu_dataset/"):
        super().__init__(img_transform, gt_transform, both_transform)
        self.root = {
            "a": {
                "train": os.path.join(root, "part_A_final/train_data/"),
                "test": os.path.join(root, "part_A_final/test_data/"),
            },
            "b": {
                "train": os.path.join(root, "part_B_final/train_data/"),
                "test": os.path.join(root, "part_B_final/test_data/"),
            }
        }[part][mode]
        self.mode = mode
        self.part = part
        self.paths = glob.glob(self.root + "images/*.jpg")
        self.load_data()

    def load_data(self):
        print("******************shtu_{part}_{mode} loading******************".format(mode=self.mode, part=self.part))
        pbar = tqdm(total=len(self.paths))
        for img_path in self.paths:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                raise IOError("{} does not exist".format(img_path))
            img = Image.open(img_path).convert('RGB')
            with h5py.File(gt_path, 'r') as gt_file:
                den = np.asarray(gt_file['density'])
            self.dataset.append([img, den])
            pbar.update(1)
        pbar.close()
