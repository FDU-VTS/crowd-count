from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------- #
# Misc setting
# ---------------------------------------------- #
_C.OUTPUT_DIR = "./output_dir"
_C.DATASET = "ShanghaiTech_B"


# ---------------------------------------------- #
# Model setting
# ---------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = "Res101"
_C.MODEL.PRETRAIN = False
_C.MODEL.DEVICE = [0, 1]


# ---------------------------------------------- #
# Solver setting
# ---------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.LEARNING_RATE = 1e-5
_C.SOLVER.EPOCH_NUM = 1000


# ---------------------------------------------- #
# Train setting
# ---------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.DATA_PATH = ""
_C.TRAIN.BATCH_SIZE = 1


# ---------------------------------------------- #
# Test setting
# ---------------------------------------------- #
_C.TEST = CN()
_C.TEST.DATA_PATH = ""
_C.TEST.BATCH_SIZE = 1

