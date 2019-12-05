# -*- coding:utf-8 -*-
import os
import glob
import skimage.io
from scipy import io
import h5py
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.io
import math
import numpy as np


# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    num = pts.shape[0] - 1
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[math.floor(pt[1]), math.floor(pt[0])] = 1.
        # if gt_count > 1:
        #     sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        # else:
        #     sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, 15, mode='constant')
    print('done.')
    return density


def shtu(root):
    # now generate the ShanghaiA's ground truth
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
    path_sets = [test_path]

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
            with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:
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

# ucf_qnrf("/home/vts/chensongjian/CrowdCount/crowd_count/data/datasets/UCF-QNRF_ECCV18_sigma_15")
