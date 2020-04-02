import sys
sys.path.append("../")
from crowdcount.data.data_preprocess import PreProcess

PreProcess(root="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/UCF-QNRF_ECCV18", name="ucf_qnrf").process()
# PreProcess(root="/home/vts/chensongjian/CrowdCount/crowdcount/data/datasets/shtu_dataset", name="shtu_b").process()