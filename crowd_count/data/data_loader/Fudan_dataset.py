from torch.utils.data import Dataset
import os
import glob
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import os.path as osp
import h5py
import numpy as np

class FudanDataset(Dataset):

    def __init__(self, mode="train", **kwargs):
        self.imgs = []
        self.root = "/home/pp/FDST_dataset/train_data/" if mode == "train" else \
                "/home/pp/FDST_dataset/test_data/"
        for root, dirs, files in os.walk(self.root, topdown=False):
            for name in dirs:
                self.imgs += glob.glob((os.path.join(root, name, "*.jpg")))
        #
        #
        # # self.paths = glob.glob(self.root + "images/*.jpg")
        # # if mode == "train":BB
        # #     self.paths *= 4
        self.transform = kwargs['transform']
        self.length = len(self.imgs)
        # # self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        path = self.imgs[item]
        img, den = self.load_data(path)
        if self.transform is not None:
            img = self.transform(img)
        # if self.transform is not None:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)

        return img, den, den

    def load_data(self, img_path):

        got_img = False

        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        if not osp.exists(img_path) or not osp.exists(gt_path):
            raise IOError("{} does not exist".format(img_path))

        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                gt_file = h5py.File(gt_path)
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass

        den = np.asarray(gt_file['density'])
        h = den.shape[0]
        w = den.shape[1]
        h_trans = h // 8
        w_trans = w // 8
        # den = cv2.resize(den, (0, 0),fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR) * 64
        # den = cv2.resize(den, (w_trans, h_trans), interpolation=cv2.INTER_LINEAR) * 64
        return img, den*1000

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = FudanDataset("train", transform=transform)
    print(dataset.__len__())
    img, den = dataset.__getitem__(500)
    print(img.size)
    print(den.shape)
