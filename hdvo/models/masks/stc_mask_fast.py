'''
Author: Ziming Liu
Date: 2022-06-30 17:11:29
LastEditors: Ziming Liu
LastEditTime: 2023-08-06 02:28:03
Description: ...
Dependent packages: don't need any extral dependency
'''
from abc import ABCMeta, abstractmethod
import enum
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from ...core import tensor_img_denorm

from hdvo.models.losses import ZNCC, ZNCCLoss

from ..builder import build_loss

from ..registry import MASKS

from hdvo.core.visulization import *
import os

from hdvo.models.utils.stereo_warping import *
from hdvo.models.utils.temporal_warping import *

@MASKS.register_module()
class STCMaskFast(nn.Module):
    def __init__(self, alpha=0.2, num_views=1, num_frames=2,  num_level=1, default_view="left", error_metric = [dict(type="L1Loss", ratio=1),
                                                                  dict(type="SSIMLoss", ratio=1)], 
                                                threshold_type="abs", mask_percent=None, threshold_ratio = 20, simple_occlu=False, old_api=False):
        '''
        description: we directly use predefined loss function to compute error, remember to let loss ratio=1. 
        return: {*}
        '''                                
        super().__init__()
        self.alpha = alpha
        self.num_views = num_views
        self.num_frames = num_frames
        self.num_level = num_level
        self.simple_occlu = simple_occlu
        self.old_api = old_api
        self.mask_percent= mask_percent
        if self.num_views == 1:
            self.views = [default_view]
        else:
            self.views = ["left", "right"]
        
        self.error_metric = error_metric
        self.threshold_type = threshold_type
        self.threshold_ratio = threshold_ratio
        self.error_func_list = [build_loss(error_metric[i]) for i in range(len(error_metric))]

    def _mask_meansurement(self, a, b):
        errors = []
        for i, error_func in enumerate(self.error_func_list):
            errors.append( self.error_func_list[i](a, b) )
        return torch.mean(torch.stack(errors), 0)

    def _forward(self, target_img, stereo_source_img, stereo_reference_disparity,
                  stereo_direction, temporal_source_img, temporal_reference_depth, T, K, invK,
                  stereo_warped_target=None, temporal_warped_target=None):
        b, c, h, w = target_img.shape
        if stereo_warped_target is None or temporal_warped_target is None:
            stereo_warped_target = stereo_warp(stereo_source_img, stereo_reference_disparity, stereo_direction)
            temporal_warped_target = temporal_warp_core(temporal_source_img, temporal_reference_depth, T, K, invK)
        non_zero_mask = (stereo_warped_target!=0) & (temporal_warped_target!=0)
        #stereo_gt_diff = self._mask_meansurement( stereo_warped_target, target_img )
        #temporal_gt_diff = self._mask_meansurement(temporal_warped_target, target_img )
        stereo_temporal_diff = self._mask_meansurement(stereo_warped_target, temporal_warped_target)
        #stereo_temporal_diff = torch.sigmoid(stereo_temporal_diff)
        #print(stereo_temporal_diff)
        possible_occlu_mask =  torch.zeros((b,1,h,w), device=target_img.device)
        possible_occlu_mask[(stereo_temporal_diff<self.threshold_ratio)] = 1
        possible_occlu_mask[torch.logical_not(non_zero_mask)] = 0
        return possible_occlu_mask.detach()
 


    def forward(self, target_img, stereo_source_img, stereo_reference_disparity,
                  stereo_direction, temporal_source_img, temporal_reference_depth, T, K, invK,
                  stereo_warped_target=None, temporal_warped_target=None):
 
        return self._forward(target_img, stereo_source_img, stereo_reference_disparity,
                  stereo_direction, temporal_source_img, temporal_reference_depth, T, K, invK,
                  stereo_warped_target, temporal_warped_target)


@MASKS.register_module()
class STCMaskFast2(nn.Module):
    def __init__(self, alpha=0.2, num_views=1, num_frames=2,  num_level=1, default_view="left", error_metric = [dict(type="ZNCCLoss", ratio=1), ], 
            threshold_type="abs", mask_percent=None, threshold_ratio = 0.8, simple_occlu=False, old_api=False):
        '''
        description: we directly use predefined loss function to compute error, remember to let loss ratio=1. 
        return: {*}
        '''                                
        super().__init__()
        self.alpha = alpha
        self.num_views = num_views
        self.num_frames = num_frames
        self.num_level = num_level
        self.simple_occlu = simple_occlu
        self.old_api = old_api
        self.mask_percent= mask_percent
        if self.num_views == 1:
            self.views = [default_view]
        else:
            self.views = ["left", "right"]
        
        self.error_metric = error_metric
        self.threshold_type = threshold_type
        self.threshold_ratio = threshold_ratio
        self.error_func_list = [build_loss(error_metric[i]) for i in range(len(error_metric))]

    def _mask_meansurement(self, a, b):
        errors = []
        for i, error_func in enumerate(self.error_func_list):
            errors.append( self.error_func_list[i](a, b) )
        return torch.mean(torch.stack(errors), 0)

    def _forward(self,  stereo_warped_target=None, temporal_warped_target=None):
        b, c, h, w = stereo_warped_target.shape
        non_zero_mask = (stereo_warped_target.sum(1,True)!=0) & (temporal_warped_target.sum(1,True)!=0)
        #stereo_gt_diff = self._mask_meansurement( stereo_warped_target, target_img )
        #temporal_gt_diff = self._mask_meansurement(temporal_warped_target, target_img )
        stereo_temporal_diff = self._mask_meansurement(stereo_warped_target, temporal_warped_target)
        #stereo_temporal_diff = torch.sigmoid(stereo_temporal_diff)
        #print(stereo_temporal_diff)
        possible_occlu_mask =  torch.zeros((b,1,h,w), device=stereo_warped_target.device)
        possible_occlu_mask[(stereo_temporal_diff<self.threshold_ratio)] = 1
        possible_occlu_mask[torch.logical_not(non_zero_mask)] = 0
        return possible_occlu_mask.detach()
 


    def forward(self,  stereo_warped_target=None, temporal_warped_target=None):
        if not isinstance(stereo_warped_target, (list,tuple)):
            stereo_warped_target = [stereo_warped_target]
        if not isinstance(temporal_warped_target, (list,tuple)):
            temporal_warped_target = [temporal_warped_target]
        assert len(stereo_warped_target) == len(temporal_warped_target)
        occlu_masks = []
        for i in range(len(stereo_warped_target)):
            occlu_masks.append(self._forward(stereo_warped_target[i], temporal_warped_target[i]))
        return occlu_masks

    def loss(self, losses, mask):
        if not isinstance(losses, (list,tuple)):
            losses = [losses]
        for i in range(len(losses)):
            for k, v in losses[i].items():
                losses[i][k] = losses[i][k] * mask[i].float()
        return losses
        