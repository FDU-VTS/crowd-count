# -*- coding:utf-8 -*-
import os
import glob
from PIL import Image
from scipy import io
import h5py
import numpy as np
from .gaussian_filter import adaptive_gaussian, uniform_gaussian


class PreProcess:
    """generate density map

    Args:
        root(str) :the root of dataset, only support shtu_a, shtu_b, ucf_qnrf and ucf_cc_50 now.
            For ShanghaiTech part A and part B, the root should be upper dir over "part_A_final" or
            "part_B_final". And for UCF QNRF and UCF CC 50, the root should be upper dir over "Train" and "Test"
        name(str, optional): the name of dataset, must be in ["shtu_a", "shtu_b", "ucf_qnrf", "ucf_cc"]. default: shtu_a

    """
    def __init__(self, root, name="shtu_a"):
        self.root = root
        self.name = name
        if self.name not in ["shtu_a", "shtu_b", "ucf_qnrf", "ucf_cc"]:
            print("the name should be 'shtu_a', 'shtu_b', 'ucf_qnrf' or 'ucf_cc'")
            raise ValueError
        self.path_set = {
            "shtu_a": [
                os.path.join(root, 'part_A_final/train_data', 'images'),
                os.path.join(root, 'part_A_final/test_data', 'images')
            ],
            "shtu_b": [
                os.path.join(root, 'part_B_final/train_data', 'images'),
                os.path.join(root, 'part_B_final/test_data', 'images')
            ],
            "ucf_qnrf": [
                os.path.join(root, 'Train'),
                os.path.join(root, 'Test')
            ],
            "ucf_cc": [root]
        }[name]

    def process(self, mode="uniform", sigma=4, radius=7):
        """generate density map and save.

        Args:
            mode(str, optional): the way to generate gaussian filter. "uniform": uniform sigma and radius of filter, the
                suggestion from C-3-Framework is sigma 15 and radius 7 as the default. "adaptive_kdtree": use adaptive
                gaussian kernel with kdtree. "adaptive_voronio": use adaptive gaussian kernel with voronio map. defalut:
                uniform.
            sigma(int, optional): the sigma of gaussian filter. default: 4.
            radius(int, optional): the radius of gaussian area. default: 7.
        """
        for path in self.path_set:
            for img_path in glob.glob(os.path.join(path, '*.jpg')):
                img = np.asarray(Image.open(img_path))
                if self.name in ["shtu_a", "shtu_b"]:
                    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images/IMG_', 'ground_truth/GT_IMG_'))
                    gt = mat["image_info"][0, 0][0, 0][0]
                    save_dir = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
                else:
                    mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
                    gt = mat["annPoints"]
                    save_dir = img_path.replace('.jpg', '.h5')
                h, w = img.shape[:2]
                density = np.zeros((h, w))
                for y, x in gt:
                    if y < h and x < w:
                        density[y, x] = 1
                if mode == "uniform":
                    density = uniform_gaussian(density, sigma=sigma, radius=radius)
                elif mode == "adaptive":
                    density = adaptive_gaussian(density, mode=mode)
                with h5py.File(save_dir, 'w') as hf:
                    hf['density'] = density
