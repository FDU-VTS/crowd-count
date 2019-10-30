# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import random
__all__ = ["ResizeShrink", "LabelEnlarge", "TransposeFlip", ]


# den: density map
# size: the size of shrinking
# zoom out the density map with cubic spline interpolation
class ResizeShrink(object):

    def __call__(self, den, size):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        w, h = den.size
        h_trans = h // size
        w_trans = w // size
        den = np.asarray(den.resize((w_trans, h_trans), Image.BICUBIC)) * (h * w) / (h_trans * w_trans)
        return den

    def __repr__(self):
        return __class__.__name__ + '()'


class LabelEnlarge(object):

    def __init__(self, number=100):
        self.number = number

    def __call__(self, den):
        return den * self.number

    def __repr__(self):
        return __class__.__name__ + '()'


class TransposeFlip(object):

    def __call__(self, img, den):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            den = den.transpose(Image.FLIP_LEFT_RIGHT)
        return img, np.asarray(den)

    def __repr__(self):
        return __class__.__name__ + '()'
