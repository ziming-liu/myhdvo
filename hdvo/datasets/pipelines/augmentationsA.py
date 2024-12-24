'''
Author: Ziming Liu
Date: 2023-02-13 15:47:39
LastEditors: Ziming Liu
LastEditTime: 2023-04-27 02:05:12
Description: ...
Dependent packages: don't need any extral dependency
'''

import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
import torch
from ..registry import PIPELINES
from ...core.visulization import vis_img_tensor
import cv2
import albumentations as A

np.random.seed(666)
random.seed(666)

# quick package for KITTI 
@PIPELINES.register_module()
class Augment_KITTI(object):
    def __init__(self, keys=[],crop_size=(352,704) ):
        self.keys = keys
        self.tran = A.ReplayCompose([A.HorizontalFlip(),
                                    A.RandomCrop(crop_size[0], crop_size[1]), # height, width
                                    ])
        self.color = A.ReplayCompose([A.RandomBrightnessContrast(),
                                    A.RandomGamma(),
                                    A.HueSaturationValue()])
    
    def __call__(self, results):
        inited = False
        replay_data=None
        for k in self.keys:
            for t in range(len(results[k])):
                if not inited:
                    replay_data = self.tran(image=results[k][t])
                    results[k][t] = replay_data["image"]
                    inited = True
                else:
                    results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
        inited = False
        replay_data=None
        for k in self.keys:
            if "imgs" in k:
                for t in range(len(results[k])):
                    if not inited:
                        replay_data = self.color(image=results[k][t])
                        results[k][t] = replay_data["image"]
                        inited = True
                    else:
                        results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
        return results

@PIPELINES.register_module()
class RandomCropA(object):
    def __init__(self, crop_size, keys=[],):
        self.keys = keys
        self.crop_size = crop_size # (h,w)
        self.tran = A.ReplayCompose([A.RandomCrop(self.crop_size[0], self.crop_size[1])])
    
    def __call__(self, results):
        inited = False
        replay_data=None
        for k in self.keys:
            for t in range(len(results[k])):
                if not inited:
                    replay_data = self.tran(image=results[k][t])
                    results[k][t] = replay_data["image"]
                    inited = True
                else:
                    results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
            
        return results



@PIPELINES.register_module()
class HorizontalFlipA(object):
    def __init__(self, keys=[], p=0.5):
        self.keys = keys
        self.p = p  
        self.tran = A.ReplayCompose([A.HorizontalFlip(p=self.p)])
    
    def __call__(self, results):
        inited = False
        replay_data=None
        for k in self.keys:
            for t in range(len(results[k])):
                if not inited:
                    replay_data = self.tran(image=results[k][t])
                    results[k][t] = replay_data["image"]
                    inited = True
                else:
                    results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
            
        return results

@PIPELINES.register_module()
class RandomBrightnessContrastA(object):
    def __init__(self, keys=[],brightness_limit=0.2,contrast_limit=0.2,
                brightness_by_max=True, always_apply=False, p=0.5):
        self.keys = keys
        self.p = p  
        self.tran = A.ReplayCompose([A.RandomBrightnessContrast(brightness_limit=brightness_limit,
         contrast_limit=contrast_limit, brightness_by_max=brightness_by_max, always_apply=always_apply,
          p=self.p)])
    
    def __call__(self, results):
        inited = False
        replay_data=None
        for k in self.keys:
            for t in range(len(results[k])):
                if not inited:
                    replay_data = self.tran(image=results[k][t])
                    results[k][t] = replay_data["image"]
                    inited = True
                else:
                    results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
            
        return results

@PIPELINES.register_module()
class RandomGammaA(object):
    def __init__(self, keys=[],gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5):
        self.keys = keys
        self.p = p  
        self.tran = A.ReplayCompose([A.RandomGamma(gamma_limit=gamma_limit, eps=eps, always_apply=always_apply, 
          p=self.p)])
    
    def __call__(self, results):
        inited = False
        replay_data=None
        for k in self.keys:
            for t in range(len(results[k])):
                if not inited:
                    replay_data = self.tran(image=results[k][t])
                    results[k][t] = replay_data["image"]
                    inited = True
                else:
                    results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
            
        return results

@PIPELINES.register_module()
class HueSaturationValueA(object):
    def __init__(self, keys=[],hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                 always_apply=False, p=0.5):
        self.keys = keys
        self.p = p  
        self.tran = A.ReplayCompose([A.HueSaturationValue(hue_shift_limit=hue_shift_limit, 
                    sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit, 
                    always_apply=always_apply, p=p)])
    
    def __call__(self, results):
        inited = False
        replay_data=None
        for k in self.keys:
            for t in range(len(results[k])):
                if not inited:
                    replay_data = self.tran(image=results[k][t])
                    results[k][t] = replay_data["image"]
                    inited = True
                else:
                    results[k][t] = A.ReplayCompose.replay(replay_data['replay'], image=results[k][t])["image"]
            
        return results