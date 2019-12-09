# -*- coding:utf-8 -*-
import os
import glob
import skimage.io
from scipy import io
import h5py
import numpy as np
from .gaussian_filter import gaussian_filter_density


def shtu(root):
    part_a_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_a_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_b_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_b_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_a_train, part_a_test, part_b_train, part_b_test]

    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            print(img_path)
            mat = io.loadmat(
                img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
            img = skimage.io.imread(img_path)
            k = np.zeros((img.shape[0], img.shape[1]))
            gt = mat["image_info"][0, 0][0, 0][0]
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            for i in range(4):
                pass
            with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
                hf['density'] = k


def ucf_qnrf(root):
    train_path = os.path.join(root, 'Train')
    test_path = os.path.join(root, 'Test')
    path_sets = [train_path]

    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            print(img_path)
            mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
            img = skimage.io.imread(img_path, plugin='matplotlib')
            k = np.zeros((img.shape[0], img.shape[1]))
            gt = mat["annPoints"]
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            with h5py.File(img_path.replace('.jpg', '.h5'), 'a') as hf:
                hf['density'] = k


def ucf_cc(root):
    for i, img_path in enumerate(glob.glob(os.path.join(root, '*.jpg'))):
        print(str(i) + " / 50")
        mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
        img = skimage.io.imread(img_path, plugin='matplotlib')
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["annPoints"]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = k


def shtu_mask(root):
    # now generate the ShanghaiA's ground truth
    part_a_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_a_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_b_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_b_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_a_train, part_a_test, part_b_train, part_b_test]

    for path in path_sets:
        img_paths = glob.glob(os.path.join(path, '*.jpg'))
        for img_path in img_paths:
            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
            mask_path = img_path.replace('images', 'masks').replace('jpg', 'h5')
            with h5py.File(mask_path) as mask_file:
                mask = np.asarray(mask_file['mask'])
            img = skimage.io.imread(img_path)
            k = np.zeros((img.shape[0], img.shape[1]))
            gt = mat["image_info"][0, 0][0, 0][0]
            for x, y in gt:
                x = int(round(x))
                y = int(round(y))
                if x < img.shape[0] and y < img.shape[1] and mask[x, y] == 0:
                    k[y, x] = 1
            k = gaussian_filter_density(k)
            with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth').replace('IMG_', 'GT_MASK_'), 'w') as hf:
                hf['density'] = k
