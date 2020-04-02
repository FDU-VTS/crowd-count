import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import math


# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def adaptive_gaussian(gt, mode="kdtree"):
    """

    Args:
        gt (numpy.ndarray): the ground truth to be processed by gaussian filter
        mode (str, optional): the way to generate density map. "uniform":

    Return:
        numpy.ndarray

    """
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[math.floor(pt[1]), math.floor(pt[0])] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


def uniform_gaussian(gt, sigma=15, radius=4):
    """

    Args:
        gt (numpy.ndarray): the ground truth to be processed by gaussian filter
        sigma (scalar or sequence of scalars, optional): Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        radius (int, optional): the radius of gaussian kernel

    Return:
        numpy.ndarray
    """
    assert isinstance(radius, int)
    assert isinstance(gt, np.ndarray)
    points = list(zip(*np.nonzero(gt)))
    h, w = gt.shape
    density = np.zeros((h, w))
    for point_h, point_w in points:
        if point_h > h or point_w > w or point_h < 0 or point_w < 0:
            continue
        h1, h2, w1, w2 = max(0, point_h - radius), min(point_h + radius + 1, h), max(0, point_w - radius), min(point_w + radius + 1, w)
        window = np.zeros((h2 - h1, w2 - w1))
        window[point_h - h1, point_w - w1] = 1
        density[h1: h2, w1: w2] += gaussian_filter(window, sigma=sigma)

    return density
