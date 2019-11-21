# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import random
import collections
import numbers

__all__ = ["SingleCompose", "ComplexCompose", "ResizeShrink",
           "LabelEnlarge", "TransposeFlip", "RandomCrop",
           "Scale"]


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

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, den):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        w, h = den.size
        h_trans = h // self.scale_factor
        w_trans = w // self.scale_factor
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

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, den):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        width, height = img.size
        h, w = self.size
        height_start = int(random.random() * (height - h))
        width_start = int(random.random() * (width - w))
        img = img.crop((width_start, height_start, width_start + w, height_start + h))
        den = den.crop((width_start, height_start, width_start + w, height_start + h))
        return img, np.asarray(den)

    def __repr__(self):
        return __class__.__name__ + '()'


class Test(object):

    def __init__(self):
        pass


class Scale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, den):
        if not isinstance(den, Image.Image):
            den = Image.fromarray(den)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img, np.asarray(den)
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation), np.asarray(den.resize((ow, oh), self.interpolation))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation), np.asarray(den.resize((ow, oh), self.interpolation))
        else:
            return img.resize(self.size[::-1], self.interpolation), \
                   np.asarray(den.resize(self.size[::-1], self.interpolation))

    def __repr__(self):
        return __class__.__name__ + '()'
