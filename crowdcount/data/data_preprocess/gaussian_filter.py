import numpy as np
import scipy
import math

# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    """

    Args:
        gt (numpy.ndarray): the ground truth to be processed by gaussian filter

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
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    num = pts.shape[0] - 1
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[math.floor(pt[1]), math.floor(pt[0])] = 1.
        # if gt_count > 1:
        #     sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        # else:
        #     sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, 15, mode='constant')
    print('done.')
    return density
