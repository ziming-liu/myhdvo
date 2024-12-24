from genericpath import exists
from numpy.testing._private.nosetester import NoseTester
from numpy.core.fromnumeric import clip
from numpy.lib.twodim_base import mask_indices
import torch
from torch.nn import functional as F
import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.shape_base import expand_dims
import cv2
import os
from hdvo.core.tensor_ops import tensor_img_denorm

def disp_map(disp):
    """
    Based on color histogram, convert the gray disp into color disp map.
    The histogram consists of 7 bins, value of each is e.g. [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    Accumulate each bin, named cbins, and scale it to [0,1], e.g. [0.114, 0.299, 0.413, 0.587, 0.701, 0.886, 1.0]
    For each value in disp, we have to find which bin it belongs to
    Therefore, we have to compare it with every value in cbins
    Finally, we have to get the ratio of it accounts for the bin, and then we can interpolate it with the histogram map
    For example, 0.780 belongs to the 5th bin, the ratio is (0.780-0.701)/0.114,
    then we can interpolate it into 3 channel with the 5th [0, 1, 0] and 6th [0, 1, 1] channel-map
    Inputs:
        disp: numpy array, disparity gray map in (Height * Width, 1) layout, value range [0,1]
    Outputs:
        disp: numpy array, disparity color map in (Height * Width, 3) layout, value range [0,1]
    """
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp


def disp_to_color(disp, max_disp=None):
    """
    Transfer disparity map to color map
    Args:
        disp (numpy.array): disparity map in (Height, Width) layout, value range [0, 255]
        max_disp (int): max disparity, optionally specifies the scaling factor
    Returns:
        disparity color map (numpy.array): disparity map in (Height, Width, 3) layout,
            range [0,255]
    """
    # grab the disp shape(Height, Width)
    h, w = disp.shape

    # if max_disp not provided, set as the max value in disp
    if max_disp is None:
        max_disp = np.max(disp)

    # scale the disp to [0,1] by max_disp
    disp = disp / max_disp

    # reshape the disparity to [Height*Width, 1]
    disp = disp.reshape((h * w, 1))

    # convert to color map, with shape [Height*Width, 3]
    disp = disp_map(disp)

    # convert to RGB-mode
    disp = disp.reshape((h, w, 3))
    disp = disp * 255.0

    return disp
def vis_img_tensor(batch_img_tensor, write_in_folder, name=None, ifnorm=True,):
    """
    author: ziming liu
    only support pytorch tensor,
    require opencv-python, pytorch
    """
    if not os.path.exists(write_in_folder):
        os.makedirs(write_in_folder)
    if len(batch_img_tensor.shape) ==3:
        batch_img_tensor = batch_img_tensor.unsqueeze(0)
    if ifnorm:
        batch_img_tensor = tensor_img_denorm(batch_img_tensor)
    batch,c,h,w = batch_img_tensor.shape
    for bidx in range(batch):
        single_img = batch_img_tensor[bidx].permute(1,2,0)
        single_img = single_img.detach().cpu().numpy()
        #single_img = cv2.normalize(single_img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        if name is not None:
            cv2.imwrite(os.path.join(write_in_folder,"{}_bs{}.png".format(name,bidx)), single_img)
        else:
            cv2.imwrite(os.path.join(write_in_folder,"bs{}.png".format(bidx)), single_img)

def vis_depth_tensor(batch_depth_tensor, write_in_folder,name=None, no_norm=False):
    """
    author: ziming liu
    only support pytorch tensor,
    require opencv-python, pytorch
    """
    if not os.path.exists(write_in_folder):
        os.makedirs(write_in_folder)
    if len(batch_depth_tensor.shape) ==4:
        batch_depth_tensor = batch_depth_tensor.squeeze(1) # remove channle dimension
    if len(batch_depth_tensor.shape)==2:
        batch_depth_tensor = batch_depth_tensor.unsqueeze(0)
    assert len(batch_depth_tensor.shape) == 3,"depth shouldn't have the channel dimension"
    batch,h,w = batch_depth_tensor.shape
    for bidx in range(batch):
        single_depth = np.array(batch_depth_tensor[bidx].float().data.cpu().numpy())
        gray_single_depth =  cv2.normalize(single_depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        if no_norm:
            gray_single_depth = single_depth
        #if name is not None:
        #    cv2.imwrite(os.path.join(write_in_folder, "{}_i{}_gray_depth_.png".format(name,bidx)), gray_single_depth)
        #else:
        #    cv2.imwrite(os.path.join(write_in_folder, "i{}_gray_depth.png".format(bidx)), gray_single_depth)

        color_depth = disp_to_color(single_depth)
        color_depth =  cv2.normalize(color_depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        if no_norm:
            color_depth = single_depth
        if name is not None:
            cv2.imwrite(os.path.join(write_in_folder, "{}_bs{}.png".format(name,bidx)), color_depth)
        else:
            cv2.imwrite(os.path.join(write_in_folder, "bs{}.png".format(bidx)), color_depth)

