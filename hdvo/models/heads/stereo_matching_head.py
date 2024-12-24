'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-07-06 14:09:51
LastEditors: Ziming Liu
LastEditTime: 2023-10-08 20:53:37
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings
import torch.distributed as dist
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from ..builder import build_backbone, build_neck, build_disp_predictor,build_loss,build_head

from ..registry import HEADS

from ..losses import DispL1Loss
 
from ...core.visulization import vis_depth_tensor,vis_img_tensor
from ..utils.inverse_warp_3d import inverse_warp_3d
import time 
from ..utils.temporal_warping import temporal_warp_c2r,  temporal_warp_r2c, temporal_warp_core
from ..utils.stereo_warping import stereo_warp_r2l, stereo_warp_l2r


@HEADS.register_module()
class StereoMatchingHead(nn.Module):
    def __init__(self, 
                 photo_loss = None,
                 struct_loss = None,
                 grid_sample_type="pytorch", 
                 padding_mode="zeros", ):
        super(StereoMatchingHead, self).__init__()
        self.grid_sample_type = grid_sample_type
        self.padding_mode = padding_mode
        if photo_loss is not None: 
            self.photo_loss = build_loss(photo_loss)
        else:
            self.photo_loss = None
        if struct_loss is not None:
            self.struct_loss = build_loss(struct_loss)
        else:
            self.struct_loss = None

    def forward(self, source_imgs, target_disps, target_imgs, direction="r2l"):

        if direction=="r2l":
            right_imgs, left_disps, left_imgs = source_imgs, target_disps, target_imgs
            warped_resutls = []
            
            warped = stereo_warp_r2l(right_imgs, left_disps, \
                                        padding_mode=self.padding_mode, \
                                        grid_sample_type=self.grid_sample_type,\
                                        gt_map=left_imgs ) # B*T C H W
            mask = (warped != 0).all(dim=1, keepdim=True).long()
        
        if direction=="l2r":
            left_imgs, right_disps, right_imgs = source_imgs, target_disps, target_imgs
             
            warped = stereo_warp_l2r(left_imgs, right_disps, \
                                        padding_mode=self.padding_mode, \
                                        grid_sample_type=self.grid_sample_type,\
                                        gt_map=right_imgs ) # B*T C H W
            mask = (warped != 0).all(dim=1, keepdim=True).long()
        #unvalid_mask_leftview = (stereo_warping_res_leftview != 0).all(dim=1, keepdim=True).long()
        
        #vis_depth_tensor(stereo_warping_res_leftview, "/home/ziliu/vis/", "leftwarping")
        loss =  self.loss(warped, target_imgs)
        loss = {k: v*mask for k,v in loss.items()}
        return loss
    
    def loss(self, warped, gt, direction="r2l"):
 
        loss = dict()
        if self.photo_loss is not None:
            photo_loss_stereo_Lview = self.photo_loss(warped, gt)
            loss[f"photo_stereo_{direction}SM"] = photo_loss_stereo_Lview
        if self.struct_loss is not None:
            struct_loss_stereo_Lview = self.struct_loss(warped, gt)
            loss[f"struct_stereo_{direction}SM"] = struct_loss_stereo_Lview
        return loss
        