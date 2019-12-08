# -*- coding:utf-8 -*-
import torch
import os


class Saver:
    """Saves the best models

    Args:
        mode (string, optional): Specifies the mode to confirm how to save the models.
            'replace' | 'remain'.
            'replace': only the best model will be saved
            'remain': the old best model won't be replaced by the new best model.
            Default: "replace"
        path (string, optional): The directory you want to save to. The default is None,
            and the sys will create a directory called "./exp" automatically.
    """
    def __init__(self, mode="remain", path=None):
        self.mode = mode
        self.path = path
        if self.path is None:
            if not os.path.exists("./exp"):
                os.mkdir("exp")
            self.path = "./exp"
        elif not os.path.exists(self.path):
            os.mkdir(self.path)

    def save(self, model, name):
        if self.mode == "replace":
            torch.save(model.state_dict(), "best_model.pt")
        elif self.mode == "remain":
            torch.save(model.state_dict(), os.path.join(self.path, "{0}.pt".format(name)))


