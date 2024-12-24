'''
Author: Ziming Liu
Date: 2022-06-30 17:11:29
LastEditors: Ziming Liu
LastEditTime: 2024-01-29 19:35:29
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

from ..registry import HEADS,MASKS

from hdvo.core.visulization import *
import os

from hdvo.models.utils.stereo_warping import *
from hdvo.models.utils.temporal_warping import *
 
try:
    from ultralytics import SAM
except:
    warnings("Not intall ultralytics")

try:
    import sys
    sys.path.append("/home/ziliu/mydata/segment-anything")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except:
    warnings("Not intall tinysam")


@MASKS.register_module()
class STCMaskSAM2(nn.Module):
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

        model_type = "vit_l"
        sam_checkpoint = "/home/ziliu/mydata/segment-anything/sam_vit_l_0b3195.pth"#sam_vit_h_4b8939.pth"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        mobile_sam.mask_threshold = 0
        print("threshold", mobile_sam.mask_threshold)
        self.predictor = SamPredictor(mobile_sam)

    def _mask_meansurement(self, a, b):
        errors = []
        for i, error_func in enumerate(self.error_func_list):
            errors.append( self.error_func_list[i](a, b) )
        return torch.mean(torch.stack(errors), 0)

    def _forward(self,  stereo_warped_target=None, temporal_warped_target=None, frame_names=None):
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

        final_masks =[]
        #x = tensor_img_denorm(x, mean=[88.78708011161852, 93.43778497818349, 91.33551888646076], std=[80.93941240862273, 81.55742718042109, 82.55097977909143])
        for i in range(stereo_warped_target.shape[0]):
            #image = x[i] # 3 h w
            image = cv2.imread(frame_names[i])
            image = cv2.resize(image, (1024,1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
            #mask = (lam_mask[i]==0).float()
            mask = -stereo_temporal_diff[i] + 0.5
            mask = F.interpolate(mask.unsqueeze(0), size=(1024,1024), mode="bilinear").squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), scale_factor=1/4, mode="bilinear").squeeze(0)
            vis_depth_tensor(mask, "/home/ziliu/vis/sam2_stc", "stc_mask")
            
            # Pad
            #mh, mw = mask.shape[-2:]
            #padh = 256 - mh
            #padw = 256 - mw
            #mask = F.pad(mask, (0, padw, 0, padh))
            
            input_point = np.array([[400, 400]])
            input_label = np.array([1])
            #mask[mask==1] = 10
            coords = (mask.squeeze()>0).nonzero()
            N = coords.shape[0]
            #index = torch.LongTensor(random.sample(range(N), int(N/5))).cuda()
            #print(index)
            #coords = torch.index_select(coords, 0, index)
            labels = torch.ones([coords.shape[0]])
            print(coords.shape)
            print(labels.shape)
            print(mask)
            masks, ious, low_res_masks_np = self.predictor.predict(
                point_coords=None, #coords.cpu().numpy(),
                point_labels=None, #labels.cpu().numpy(),
                box=None,
                mask_input = mask.detach().cpu().numpy(),
                multimask_output=True,
                )
            #print(low_res_masks_np)
            #print(masks.shape)
            masks = F.interpolate(torch.FloatTensor(masks).unsqueeze(0), size=(h,w), mode="nearest").squeeze(0)
            for i in range(masks.shape[0]):
                vis_depth_tensor(torch.FloatTensor(masks[i]), "/home/ziliu/vis/sam2_stc", f"mask{i}")
            new_seg = np.array(masks[np.argmax(ious)],order="F", dtype="uint8")
            vis_depth_tensor(torch.FloatTensor(new_seg), "/home/ziliu/vis/sam2_stc", f"finalmask")
            final_masks.append(new_seg)

        return possible_occlu_mask.detach()

    def forward(self,  stereo_warped_target=None, temporal_warped_target=None, frame_names=None):
        if not isinstance(stereo_warped_target, (list,tuple)):
            stereo_warped_target = [stereo_warped_target]
        if not isinstance(temporal_warped_target, (list,tuple)):
            temporal_warped_target = [temporal_warped_target]
        assert len(stereo_warped_target) == len(temporal_warped_target)
        occlu_masks = []
        for i in range(len(stereo_warped_target)):
            occlu_masks.append(self._forward(stereo_warped_target[i], temporal_warped_target[i], frame_names))
        return occlu_masks

    def loss(self, losses, mask):
        if not isinstance(losses, (list,tuple)):
            losses = [losses]
        for i in range(len(losses)):
            for k, v in losses[i].items():
                losses[i][k] = losses[i][k] * mask[i].float()
        return losses
        

