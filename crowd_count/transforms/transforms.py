# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as F
__all__ = ["SingleCompose", "ComplexCompose", "ResizeShrink", "LabelEnlarge", "TransposeFlip", "RandomCrop"]


# den: density map
# size: the size of shrinking
# zoom out the density map with cubic spline interpolation
class SingleCompose(object):

    def __init__(self, cc_transforms):
        self.cc_transforms = cc_transforms

    def __call__(self, img):
        for t in self.cc_transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.cc_transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComplexCompose(object):

    def __init__(self, cc_transforms):
        self.cc_transforms = cc_transforms

    def __call__(self, img, den):
        for t in self.cc_transforms:
            img, den = t(img, den)
        return img, den

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.cc_transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ResizeShrink(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, den):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        w, h = den.size
        h_trans = h // self.size
        w_trans = w // self.size
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


class RandomCrop(object):

    def __init__(self, crop_height, crop_width):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __call__(self, img, den):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        height = img.size[0]
        width = img.size[1]
        height_start = int(random.random() * (height - self.crop_height))
        width_start = int(random.random() * (width - self.crop_width))
        img = F.crop(img, height_start, width_start, self.crop_height, self.crop_width)
        den = F.crop(den, height_start, width_start, self.crop_height, self.crop_width)
        return img, np.asarray(den)


class FixedCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, den):
        result = []
        den = np.asarray(den)
        img = np.asarray(img)
        height = den.shape[0]
        width = den.shape[1]
        h = height // self.crop_size
        w = width // self.crop_size
        result.append([img[: h, : w], den[: h, : w]])
        result.append([img[: h, w:], den[: h, w:]])
        result.append([img[h:, : w], den[h:, : w]])
        result.append([img[h:, w:], den[h:, w:]])
        return result
