'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-04-18 16:10:53
LastEditors: Ziming Liu
LastEditTime: 2023-04-19 02:38:00
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from zimingdepth.models.utils.inverse_warp_3d import inverse_warp_3d
import time

from torch.utils.cpp_extension import load
from multiprocessing import Pool


def build_image_volume_cuda(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=None):
    return cuda_module.forward(leftImage, rightImage, max_disp, mask_template)


def build_image_volume(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False):
    device = leftImage.device
    B, C, H, W = leftImage.shape
    D = max_disp
    disp_sample = torch.linspace(0, D-1, D, device=device)
    disp_sample = disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W)
    # expand D dimension
    concat_leftImage = leftImage.unsqueeze(2).expand(B, C, D, H, W)
    concat_rightImage = rightImage.unsqueeze(2).expand(B, C, D, H, W)

    # shift target feature according to disparity samples
    concat_rightImage = inverse_warp_3d(concat_rightImage.float(), -disp_sample.float(), padding_mode='zeros')

    # mask out features in reference
    if mask_left:
        concat_leftImage = concat_leftImage * (concat_rightImage != 0) # fix the type bug when using half-float. ziming 21-7-8
    if img_pad_zeros:
        pad_val = torch.mean(torch.cat([leftImage,rightImage], dim=1).reshape(-1))
        concat_rightImage[concat_rightImage==0] = pad_val # mean value to replace zero padding
    #concat_leftImage[concat_leftImage==0] = pad_val # mean value to replace zero padding
    #for i in range(192):
    #    vis_img_tensor(concat_rightImage[0,:,i,:,:], '/home/ziliu/vis/pixelnet/', f"concat_rightImage_{i}")
    #    vis_img_tensor(concat_leftImage[0,:,i,:,:], '/home/ziliu/vis/pixelnet/', f"concat_leftImage_{i}")
    # [B, 2C, D, H, W)
    image_volume = torch.cat((concat_leftImage, concat_rightImage), dim=1)

    return image_volume


def build_image_volume_loop(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False):
    device = leftImage.device
    B, C, H, W = leftImage.shape
    D = max_disp
    disp_sample = torch.linspace(0, D-1, D, device=device)
    #disp_sample = disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W)
    # expand D dimension
    #concat_leftImage = leftImage.unsqueeze(2).expand(B, C, D, H, W)
    #concat_rightImage = rightImage.unsqueeze(2).expand(B, C, D, H, W)

    image_volume = torch.zeros(B, C * 2, D, H, W).to(device) # fix the type bug when using half-float. ziming 21-7-8
    idx = 0
    for i in disp_sample:
        i = i.long() # convert torch.Tensor to int, so that it can be index
        if i > 0:
            image_volume[:, :C, idx, :, i:] = leftImage[:, :, :, i:]
            image_volume[:, C:, idx, :, i:] = rightImage[:, :, :, :-i]
        elif i == 0:
            image_volume[:, :C, idx, :, :] = leftImage
            image_volume[:, C:, idx, :, :] = rightImage
        else:
            image_volume[:, :C, idx, :, :i] = leftImage[:, :, :, :i]
            image_volume[:, C:, idx, :, :i] = rightImage[:, :, :, abs(i):]
        idx = idx + 1
 
    return image_volume
 


def build_image_volume_roll(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=None):
    device = leftImage.device
    B, C, H, W = leftImage.shape
    D = max_disp
    #disp_sample = torch.linspace(0, D-1, D,)
    #disp_sample = disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W)
    # expand D dimension
    #concat_leftImage = leftImage.unsqueeze(2).expand(B, C, D, H, W)
    #concat_rightImage = rightImage.unsqueeze(2).expand(B, C, D, H, W)

    #image_volume = torch.zeros(B, C * 2, D, H, W).to(device) # fix the type bug when using half-float. ziming 21-7-8
    #idx = 0
    image_volume = [torch.cat((leftImage, torch.roll(rightImage, i, dims=3)), dim=1) for i in range(D)]

    #for i in range(D):
    #    image_volume.append(torch.cat((leftImage, torch.roll(rightImage, i, dims=3)), dim=1))
    image_volume = torch.stack(image_volume, dim=2)
    
    if mask_template is not None:
        image_volume = image_volume * mask_template
    return image_volume

def build_image_volume_roll_half(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=None):
    leftImage, rightImage = leftImage.half(), rightImage.half()
    device = leftImage.device
    B, C, H, W = leftImage.shape
    D = max_disp
    #disp_sample = torch.linspace(0, D-1, D,)
    #disp_sample = disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W)
    # expand D dimension
    #concat_leftImage = leftImage.unsqueeze(2).expand(B, C, D, H, W)
    #concat_rightImage = rightImage.unsqueeze(2).expand(B, C, D, H, W)

    #image_volume = torch.zeros(B, C * 2, D, H, W).to(device) # fix the type bug when using half-float. ziming 21-7-8
    #idx = 0
    image_volume = [torch.cat((leftImage, torch.roll(rightImage, i, dims=3)), dim=1) for i in range(D)]

    #for i in range(D):
    #    image_volume.append(torch.cat((leftImage, torch.roll(rightImage, i, dims=3)), dim=1))
    image_volume = torch.stack(image_volume, dim=2)
    
    if mask_template is not None:
        mask_template = mask_template.half()
        image_volume = image_volume * mask_template
    return image_volume

def build_image_volume_roll_parallel(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=None):
    device = leftImage.device
    B, C, H, W = leftImage.shape
    D = max_disp

    def func(i):
        return torch.cat((leftImage, torch.roll(rightImage, i, dims=3)), dim=1)
    #for i in range(D):
        #i = int(i) # convert torch.Tensor to int, so that it can be index
        #if i > 0:
    image_volume_list = list(map(func, range(D)))
        #image_volume[:, :C, idx, :, :] = leftImage
        #image_volume[:, C:, idx, :, :] = torch.roll(rightImage, i, dims=3)
        #idx = idx + 1
    image_volume = torch.stack(image_volume_list, dim=2)
    if mask_template is not None:
        image_volume = image_volume * mask_template
    return image_volume

def build_image_volume_pad(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False):
    device = leftImage.device
    B, C, H, W = leftImage.shape
    D = max_disp
    #disp_sample = torch.linspace(0, D-1, D,)
    #disp_sample = disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W)
    # expand D dimension
    #concat_leftImage = leftImage.unsqueeze(2).expand(B, C, D, H, W)
    #concat_rightImage = rightImage.unsqueeze(2).expand(B, C, D, H, W)

    #image_volume = torch.zeros(B, C * 2, D, H, W).to(device) # fix the type bug when using half-float. ziming 21-7-8
    idx = 0
    image_volume = []
    for i in range(D):
        i = int(i) # convert torch.Tensor to int, so that it can be index
        #if i > 0:
        if i==0:
            image_volume.append(torch.cat((leftImage,rightImage), 1))
            continue
        image_volume.append(torch.cat((F.pad(leftImage[:, :, :, i:], (i,0),'constant',0), F.pad(rightImage[:, :, :, :-i],(i,0),'constant',0)), dim=1))
        #image_volume[:, :C, idx, :, :] = leftImage
        #image_volume[:, C:, idx, :, :] = torch.roll(rightImage, i, dims=3)

        
        idx = idx + 1
    image_volume = torch.stack(image_volume, dim=2)
    
    return image_volume


if __name__ == '__main__':
    cuda_module = load(name="build_image_volume",
                   sources=["zimingdepth/models/utils/build_image_volume/build_image_volume/build_image_volume_cuda.cpp", 
                            "zimingdepth/models/utils/build_image_volume/build_image_volume/build_image_volume_kernel.cu"],
                   verbose=True)
    #with torch.cuda.amp.autocast():
    t1 = time.time()
    mask_template = torch.ones(8, 2*3, 192, 256, 512).cuda()
    for i in range(192):
        mask_template[:, :, i, :, 0:i] = 0
    t2 = time.time()
    print("generate mask template time: ", t2-t1)
    #for i in range(192):
    #    print(mask_template[0,0, i, 0, :])
    leftImage = torch.randn(8, 3, 256, 512).cuda()
    rightImage = torch.randn(8, 3, 256, 512).cuda()
    t1 = time.time()
    image_volume = build_image_volume(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False)
    t2 = time.time()
    print("warp method 1 time: ", t2-t1)
    print(image_volume.shape)

    t1 = time.time()
    image_volume = build_image_volume_loop(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False)
    t2 = time.time()
    print("loop method 2 time: ", t2-t1)
    print(image_volume.shape)

    t1 = time.time()
    image_volume = build_image_volume_roll(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False)
    t2 = time.time()
    print("roll method 3 time: ", t2-t1)
    print(image_volume.shape)

    t1 = time.time()
    roll_mask_image_volume = build_image_volume_roll(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=mask_template)
    t2 = time.time()
    print("masked roll method 3 time: ", t2-t1)
    print(roll_mask_image_volume.shape)
    #for i in range(192):
    #    print(image_volume[0,0, i, 0, :])
    #    time.sleep(1)

    t1 = time.time()
    roll_mask_image_volume = build_image_volume_roll_half(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=mask_template)
    t2 = time.time()
    print("masked roll .half() method 3 time: ", t2-t1)
    print(roll_mask_image_volume.shape)

    t1 = time.time()
    image_volume = build_image_volume_roll_parallel(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=mask_template)
    t2 = time.time()
    print("masked roll parallel method 3 time: ", t2-t1)
    print(image_volume.shape)

    t1 = time.time()
    image_volume = build_image_volume_pad(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False)
    t2 = time.time()
    print("pad method 4 time: ", t2-t1)
    
    t1 = time.time()
    image_volume = build_image_volume_cuda(leftImage, rightImage, max_disp=192, mask_left=False, img_pad_zeros=False, mask_template=mask_template)
    t2 = time.time()
    print("cuda method 5 time: ", t2-t1)
    #print(roll_mask_image_volume==image_volume)
