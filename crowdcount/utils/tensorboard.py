import torch.utils.tensorboard as tensorboard


class TensorBoard:
    """ create logs with tensorflow/tensorboard

    Args:
        path (str, optional): the dir to store the logs (default: ./runs).
    """

    def __init__(self, path="./runs"):
        self.path = path
        self.summary_writer = tensorboard.SummaryWriter(self.path)

    def write_diagram(self, tag, value, step):
        self.summary_writer.add_scalar(tag, value, step)

    def write_model(self):
        pass
