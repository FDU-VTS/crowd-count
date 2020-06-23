import sys
sys.path.append("../")
import argparse
from crowdcount.engine import train
from crowdcount.utils import *
from crowdcount.config import cfg
from crowdcount.models import build_model
from crowdcount.data.data_loader import build_data


def main(cfg):
    model = build_model(cfg)
    train_set, test_set = build_data(cfg)
    train_loss = AVGLoss()
    test_loss = EnlargeLoss(100)
    saver = Saver(path="../exp/2019-04-02-shtu_b")
    tb = TensorBoard(path="../runs/2019-04-02-shtu_b")
    train(model,
          train_set,
          test_set,
          train_loss,
          test_loss,
          optim=cfg.SOLVER.OPTIMIZER,
          saver=saver,
          cuda_num=cfg.MODEL.DEVICE,
          train_batch=4,
          test_batch=1,
          learning_rate=1e-5,
          epoch_num=1000,
          enlarge_num=100,
          tensorboard=tb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch package for Crowd Counting")
    parser.add_argument(
        "--config",
        default="config/shtu_b_res50.yaml",
        help="path to config file",
        type=str
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    main(cfg)
