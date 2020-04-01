import sys
from pandas import read_csv
import numpy as np
from scipy import io
import scipy.ndimage
import skimage.io
sys.path.append("../")

root1 = "/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/shanghaitech_part_B/train/den/1.csv"
root2 = "/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/shtu_dataset/part_B_final/train_data/ground_truth/GT_IMG_1.mat"
img_path2 = "/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/shtu_dataset/part_B_final/train_data/images/IMG_1.jpg"
img_path1 = "/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/shanghaitech_part_B/train/img/1.jpg"

csv = read_csv(root1).values
den1 = csv.astype(np.float32, copy=False)
print("den1: ", den1.shape)

img = skimage.io.imread(img_path1)
print("img1: ", img.shape)

mat = io.loadmat(root2)
k = np.zeros((img.shape[0], img.shape[1]))
gt = mat["image_info"][0, 0][0, 0][0]
for i in range(0, len(gt)):
    if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
        k[int(gt[i][1]), int(gt[i][0])] = 1
print("gt: ", k.shape)

den2 = scipy.ndimage.gaussian_filter(k, sigma=15, mode="nearest")
print("den2: ", den2.shape)
print("sum: ", np.sum(den2), np.sum(den1), np.sum(k))
