'''
Author: Ziming Liu
Date: 2022-06-30 17:11:29
LastEditors: Ziming Liu
LastEditTime: 2023-11-17 16:18:10
Description: ...
Dependent packages: don't need any extral dependency
'''
from abc import ABCMeta, abstractmethod
import enum
import warnings

import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from ...core import tensor_img_denorm

from hdvo.models.losses import ZNCC, ZNCCLoss

from ..builder import build_loss

from ..registry import HEADS

from hdvo.core.visulization import *
import os

from hdvo.models.utils.stereo_warping import *
from hdvo.models.utils.temporal_warping import *

try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except:
    warnings("Not intall mobile_sam")

try:
    from ultralytics import SAM
except:
    warnings("Not intall ultralytics")
    
@HEADS.register_module()
class STCMaskSAM(nn.Module):
    def __init__(self, alpha=0.2, num_views=1, num_frames=2,  num_level=1, default_view="left", error_metric = [dict(type="L1Loss", ratio=1),
                                                                  dict(type="SSIMLoss", ratio=1)], 
                                                threshold_type="abs", mask_percent=None, threshold_ratio = 20, simple_occlu=False, old_api=False,
                                                mask_type="stc"):
        '''
        description: we directly use predefined loss function to compute error, remember to let loss ratio=1. 
        return: {*}
        '''                                
        super().__init__()
        self.mask_type = mask_type
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

        self.sam_model = SAM('sam_l.pt')
        #self.sam_model = SAM('mobile_sam.pt')
        #model = SAM('sam_b.pt')
        self.sam_model.info()  # display model information

        #model_type = "vit_t"
        #sam_checkpoint = "/home/ziliu/mydata/MobileSAM/weights/mobile_sam.pt"
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        #mobile_sam.to(device=device)
        #mobile_sam.eval()
        #self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        #self.predictor = SamPredictor(mobile_sam)

    def _mask_meansurement(self, a, b):
        errors = []
        for i, error_func in enumerate(self.error_func_list):
            errors.append( self.error_func_list[i](a, b) )
        return torch.mean(torch.stack(errors), 0)

    def stc_forward(self, target_img_files, stereo_source_img_files, stereo_reference_disparity,
                  stereo_direction, temporal_source_img_files, temporal_reference_depth, T, K, invK,
                  stereo_warped_target=None, temporal_warped_target=None):
        b, c, h, w = stereo_reference_disparity.shape

        stereo_source_img = []
        for frame in stereo_source_img_files:
            results = self.sam_model.predict(frame)
            results = results[0]
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0).float(), size=(h,w),mode="nearest",).squeeze(0)
            
            """ 
            input = cv2.imread(frame)
            input = cv2.resize(input, (w,h), interpolation=cv2.INTER_LINEAR)
            #t0 = time.time()
            #sam_masks = self.mask_generator.generate(input)
            self.predictor.set_image(input)
            sam_masks, _, _ = self.predictor.predict()
            sam_masks = torch.FloatTensor(sam_masks).cuda()
            #torch.cuda.synchronize()
            #t1 = time.time()
            #print("time: ", t1-t0)
            """
            stereo_source_img.append(sam_masks.cuda())
        stereo_source_img = torch.stack(stereo_source_img) # B N H W 
        
        temporal_source_img = []
        for frame in temporal_source_img_files:
            results = self.sam_model.predict(frame)
            results = results[0]
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0).float(), size=(h,w),mode="nearest",).squeeze(0)
            
            """
            input = cv2.imread(frame)
            input = cv2.resize(input, (w,h), interpolation=cv2.INTER_LINEAR)
            #t0 = time.time()
            #sam_masks = self.mask_generator.generate(input)
            self.predictor.set_image(input)
            sam_masks, _, _ = self.predictor.predict()
            sam_masks = torch.FloatTensor(sam_masks).cuda()
            #torch.cuda.synchronize()
            #t1 = time.time()
            #print("time: ", t1-t0)
            """
            temporal_source_img.append(sam_masks.cuda())
        temporal_source_img = torch.stack(temporal_source_img) # B N H W 
        
        if stereo_warped_target is None or temporal_warped_target is None:
            stereo_warped_target = stereo_warp(stereo_source_img, stereo_reference_disparity, stereo_direction)
            temporal_warped_target = temporal_warp_core(temporal_source_img, temporal_reference_depth, T, K, invK)
        
        batch_sam_stc_mask = []#torch.zeros((b,1,h,w), device=stereo_reference_disparity.device)
        for b_idx in range(b):
            res_s = stereo_warped_target[b_idx]
            res_t = temporal_warped_target[b_idx]
            N, H, W = res_s.shape
            sam_stc_mask = torch.ones((H,W), device=stereo_reference_disparity.device)
            for stereo_n_idx in range(res_s.shape[0]):
                for temporal_n_idx in range(res_t.shape[0]):
                    stc_intersection = torch.logical_and(res_s[stereo_n_idx], res_t[temporal_n_idx])
                    stc_sum = torch.logical_or(res_s[stereo_n_idx], res_t[temporal_n_idx])
                    if torch.sum(stc_intersection.long())/ torch.sum(stc_sum.long()) > 0.8:
                        #sam_stc_mask[stc_intersection] = 1
                        sam_stc_mask[stc_sum!=stc_intersection] = 0
            #vis_depth_tensor(sam_stc_mask, "/home/ziliu/vis/semantic_stc")
            batch_sam_stc_mask.append(sam_stc_mask)
        possible_occlu_mask = torch.stack(batch_sam_stc_mask).unsqueeze(1)
        return possible_occlu_mask.detach()
 
    def temporal_forward(self, target_img_files,
                 temporal_source_img_files, temporal_reference_depth, T, K, invK,
                  stereo_warped_target=None, temporal_warped_target=None):
        b, c, h, w = temporal_reference_depth.shape
        target_img = []
        for frame in target_img_files:
            results = self.sam_model.predict(frame)
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0), size=(h,w),mode="nearest",align_corners=True).squeeze(0)
            target_img.append(sam_masks)
        target_img = torch.stack(target_img) # B N H W 

        temporal_source_img = []
        for frame in temporal_source_img_files:
            results = self.sam_model.predict(frame)
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0), size=(h,w),mode="nearest",align_corners=True).squeeze(0)
            temporal_source_img.append(sam_masks)
        temporal_source_img = torch.stack(temporal_source_img) # B N H W 

        temporal_warped_target = temporal_warp_core(temporal_source_img, temporal_reference_depth, T, K, invK)
        batch_sam_stc_mask = torch.zeros((b,1,h,w), device=temporal_warped_target.device)
        for b_idx in range(b):
            res_target = target_img[b_idx]
            res_t = temporal_warped_target[b_idx]
            N, H, W = res_t.shape
            sam_stc_mask = torch.zeros((H,W), device=temporal_warped_target.device)
            for target_n_idx in range(N):
                for temporal_n_idx in range(N):
                    stc_intersection = torch.logical_and(res_target[target_n_idx], res_t[temporal_n_idx])
                    if torch.sum(stc_intersection.long()) > 0:
                        sam_stc_mask[stc_intersection] = 1
            batch_sam_stc_mask.append(sam_stc_mask)
        possible_occlu_mask = torch.stack(batch_sam_stc_mask).unsqueeze(1)
        return possible_occlu_mask.detach()

    def stereo_forward(self, target_img_files, stereo_source_img_files, stereo_reference_disparity,
                  stereo_direction, 
                  stereo_warped_target=None, temporal_warped_target=None):
        b, c, h, w = stereo_reference_disparity.shape
        target_img = []
        for frame in target_img_files:
            results = self.sam_model.predict(frame)
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0), size=(h,w),mode="nearest",align_corners=True).squeeze(0)
            target_img.append(sam_masks)
        target_img = torch.stack(target_img) # B N H W 

        stereo_source_img = []
        for frame in stereo_source_img_files:
            results = self.sam_model.predict(frame)
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0), size=(h,w),mode="nearest",align_corners=True).squeeze(0)
            stereo_source_img.append(sam_masks)
        stereo_source_img = torch.stack(stereo_source_img) # B N H W 

        stereo_warped_target = stereo_warp(stereo_source_img, stereo_reference_disparity, stereo_direction)
        batch_sam_stc_mask = torch.zeros((b,1,h,w), device=temporal_warped_target.device)
        for b_idx in range(b):
            res_s = stereo_warped_target[b_idx]
            res_target = target_img[b_idx]
            N, H, W = res_s.shape
            sam_stc_mask = torch.zeros((H,W), device=temporal_warped_target.device)
            for stereo_n_idx in range(N):
                for target_n_idx in range(N):
                    stc_intersection = torch.logical_and(res_s[stereo_n_idx], res_target[target_n_idx])
                    if torch.sum(stc_intersection.long()) > 0:
                        sam_stc_mask[stc_intersection] = 1
            batch_sam_stc_mask.append(sam_stc_mask)
        possible_occlu_mask = torch.stack(batch_sam_stc_mask).unsqueeze(1)
        return possible_occlu_mask.detach()


    def forward(self, **kwargs):
        if self.mask_type == "stc":
            return self.stc_forward(**kwargs)
        if self.mask_type == "temporal":
            return self.temporal_forward(**kwargs)
        if self.mask_type == "stereo":
            return self.stereo_forward(**kwargs)
        return 0


