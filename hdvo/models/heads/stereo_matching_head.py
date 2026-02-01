"""
Stereo Matching Head Module.

This module implements photometric and structural consistency losses for
stereo image matching through warping operations.

Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-07-06
Last Modified: 2023-10-08
"""

import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import HEADS
from ..utils.stereo_warping import stereo_warp_r2l, stereo_warp_l2r


@HEADS.register_module()
class StereoMatchingHead(nn.Module):
    """Stereo image matching head with photometric and structural losses.
    
    This head computes photometric and structural consistency losses between
    stereo image pairs through warping operations.
    
    Args:
        photo_loss (dict, optional): Configuration for photometric loss.
        struct_loss (dict, optional): Configuration for structural loss.
        grid_sample_type (str): Type of grid sampling. Defaults to 'pytorch'.
        padding_mode (str): Padding mode for grid sampling. Defaults to 'zeros'.
    """
    
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
        """Forward pass for stereo matching.
        
        Args:
            source_imgs: Source images to be warped.
            target_disps: Target disparity maps.
            target_imgs: Target images for comparison.
            direction (str): Warping direction, 'r2l' (right to left) or 
                'l2r' (left to right). Defaults to 'r2l'.
                
        Returns:
            dict: Dictionary containing computed losses.
        """

        if direction=="r2l":
            right_imgs, left_disps, left_imgs = source_imgs, target_disps, target_imgs
            warped_resutls = []
            
            warped = stereo_warp_r2l(right_imgs, left_disps, \
                                        padding_mode=self.padding_mode, \
                                        grid_sample_type=self.grid_sample_type,\
                                        gt_map=left_imgs )
            mask = (warped != 0).all(dim=1, keepdim=True).long()
        
        if direction=="l2r":
            left_imgs, right_disps, right_imgs = source_imgs, target_disps, target_imgs
             
            warped = stereo_warp_l2r(left_imgs, right_disps, \
                                        padding_mode=self.padding_mode, \
                                        grid_sample_type=self.grid_sample_type,\
                                        gt_map=right_imgs )
            mask = (warped != 0).all(dim=1, keepdim=True).long()
        
        loss =  self.loss(warped, target_imgs)
        loss = {k: v*mask for k,v in loss.items()}
        return loss
    
    def loss(self, warped, gt, direction="r2l"):
        """Compute photometric and structural losses.
        
        Args:
            warped: Warped images.
            gt: Ground truth target images.
            direction (str): Warping direction for loss naming.
            
        Returns:
            dict: Dictionary with photometric and structural loss values.
        """
        loss = dict()
        if self.photo_loss is not None:
            photo_loss_stereo_Lview = self.photo_loss(warped, gt)
            loss[f"photo_stereo_{direction}SM"] = photo_loss_stereo_Lview
        if self.struct_loss is not None:
            struct_loss_stereo_Lview = self.struct_loss(warped, gt)
            loss[f"struct_stereo_{direction}SM"] = struct_loss_stereo_Lview
        return loss
        