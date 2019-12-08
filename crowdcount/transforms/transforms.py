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
    """Compose several transforms witch only transform single input (image or density map)

    Args:
        transforms (list of ``Transform`` objects which transform one object:
            [``ResizeShrink``, ``LabelEnlarge``]): list of transforms to Compose

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> cc_transforms.SingleCompose([
        >>>     ResizeShrink(8),
        >>>     LabelEnlarge(10),
        >>> ])
    """
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
    """Compose several transforms witch transform both of image and density map

    Args:
        transforms (list of ``Transform`` objects which transform two objects:
            [``TransposeFlip``, ``RandomCrop``, ``Scale``]): list of transforms to Compose

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> cc_transforms.ComplexCompose([
        >>>     TransposeFlip(),
        >>>     RandomCrop([512, 512]),
        >>>     Scale([512, 512]),
        >>> ])
    """

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
    """ Reduce the density map scale_factor times (to suit the output be pooled)

    Args:
        scale_factor (int): Desired reduction factor. The output size will be divided by scale_factor.
            If the scale_factor is 8 and the size of input is (20, 10), the output size will be (20 // 8, 10 // 8) = (2, 1)
            to match the output image which be pooled.

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> import numpy as np
        >>> resize_shrink = cc_transforms.ResizeShrink(8)
        >>> density_map = np.random.rand(20, 10)
        >>> density_map.shape
        (20, 10)
        >>> resize_shrink(density_map)
        array([[52.175777], [46.061344]], dtype=float32)
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, den):
        """
        Args:
            density map (PIL Image or numpy.ndarray): density map to be shrunk

        Returns:
            numpy.ndarray: Rescaled image
        """
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
    """ Training trick from the
    `"C^3 Framework..." <https://arxiv.org/abs/1907.02724>`_ paper.
    They find neural network could get faster convergence and lower estimation
    error when the density map dots a large integer value

    Args:
        number (int): the magnification of density map. default is 100.

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> import numpy as np
        >>> label_enlarge = cc_transforms.LabelEnlarge(100)
        >>> density_map = np.random.rand(4, 4)
        array([[0.62494003, 0.35120895, 0.21002026, 0.52596833],
               [0.45540917, 0.41721004, 0.45287173, 0.35665398],
               [0.63187118, 0.25588367, 0.44660365, 0.0367272 ],
               [0.86808967, 0.11982928, 0.44544907, 0.81409479]])
        >>> label_enlarge(density_map)
        array([[62.49400293, 35.12089511, 21.00202647, 52.59683332],
               [45.54091666, 41.72100355, 45.28717272, 35.66539753],
               [63.18711774, 25.58836725, 44.66036518,  3.67272008],
               [86.8089669 , 11.98292805, 44.54490721, 81.40947874]])
    """

    def __init__(self, number=100):
        self.number = number

    def __call__(self, den):
        """
        Args:
            density map (PIL Image or numpy.ndarray): density map to be enlarged

        Returns:
            numpy.ndarray: enlarged density map
        """
        return den * self.number

    def __repr__(self):
        return __class__.__name__ + '()'


class TransposeFlip(object):
    """ Randomly flip both of the image and density map left and right

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> import numpy as np
        >>> img = np.randn(4, 4)
        >>> density_map = np.randn(4, 4)
        >>> transpose_flip = cc_transforms.TransposeFlip()
        >>> img, density_map = transpose_flip(img, density_map)
    """

    def __call__(self, img, den):
        """
        Args:
            image (PIL Image or numpy.ndarray): image to be flipped
            density map (PIL Image or numpy.ndarray): density map to be flipped

        Returns:
            (PIL Image, numpy.ndarray)
        """
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
    """ In order to use multi-batch training to irregular datasets (like ShanghaiTech Part A where images
    have different shape), This function random crops both of image and density map with input size.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an int instead of sequence like (h, w),
        a square crop (size, size) is made.

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> import numpy as np
        >>> img = np.randn(4, 4)
        >>> density_map = np.randn(4, 4)
        >>> random_crop = cc_transforms.RandomCrop([2, 2])
        >>> img, density_map = random_crop(img, density_map)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, den):
        """
        Args:
            image (PIL Image or numpy.ndarray): image to be cropped
            density map (PIL Image or numpy.ndarray): density map to be cropped

        Returns:
            (PIL Image, numpy.ndarray)
        """
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


class Scale(object):
    """In order to use multi-batch training to irregular datasets (like ShanghaiTech Part A where images
    have different shape), This function resize both of image and density map with input size.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an int instead of sequence like (h, w),
        a square crop (size, size) is made.

    Example:
        >>> import crowdcount.transforms as cc_transforms
        >>> import numpy as np
        >>> img = np.randn(4, 4)
        >>> density_map = np.randn(4, 4)
        >>> scale = cc_transforms.Scale([2, 2])
        >>> img, density_map = scale(img, density_map)
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, den):
        """
        Args:
            image (PIL Image or numpy.ndarray): image to be cropped
            density map (PIL Image or numpy.ndarray): density map to be cropped

        Returns:
            (PIL Image, numpy.ndarray)
        """
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
