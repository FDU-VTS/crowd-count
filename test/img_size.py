import skimage.io
import glob
import os


root = "/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/ProcessedData/UCF-qnrf/train/img"
img_paths = glob.glob(os.path.join(root, "*.jpg"))
for img_path in img_paths:
    img = skimage.io.imread(img_path)
    print(img.shape)
