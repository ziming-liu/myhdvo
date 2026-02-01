"""
Fast stereo-temporal consistency (STC) mask module.

Author: Ziming Liu
Date: 2022-06-30
Last Modified: 2023-08-06
"""
import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import MASKS
from hdvo.models.utils.stereo_warping import stereo_warp
from hdvo.models.utils.temporal_warping import temporal_warp_core

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
        """Measure error between two tensors using multiple metrics.
        
        Args:
            a (Tensor): First tensor
            b (Tensor): Second tensor
            
        Returns:
            Tensor: Mean error across all metrics
        """
        errors = [func(a, b) for func in self.error_func_list]
        return torch.mean(torch.stack(errors), 0)

    def _forward(
        self,
        target_img,
        stereo_source_img,
        stereo_reference_disparity,
        stereo_direction,
        temporal_source_img,
        temporal_reference_depth,
        T,
        K,
        invK,
        stereo_warped_target=None,
        temporal_warped_target=None
    ):
        """Internal forward pass for mask generation.
        
        Args:
            target_img (Tensor): Target image
            stereo_source_img (Tensor): Stereo source image
            stereo_reference_disparity (Tensor): Stereo disparity map
            stereo_direction (str): Stereo direction
            temporal_source_img (Tensor): Temporal source image
            temporal_reference_depth (Tensor): Temporal depth map
            T (Tensor): Transformation matrix
            K (Tensor): Camera intrinsic matrix
            invK (Tensor): Inverse camera intrinsic matrix
            stereo_warped_target (Tensor, optional): Pre-computed stereo warped image
            temporal_warped_target (Tensor, optional): Pre-computed temporal warped image
            
        Returns:
            Tensor: Occlusion mask of shape (B, 1, H, W)
        """
        b, c, h, w = target_img.shape
        
        # Perform warping if not provided
        if stereo_warped_target is None or temporal_warped_target is None:
            stereo_warped_target = stereo_warp(
                stereo_source_img, stereo_reference_disparity, stereo_direction
            )
            temporal_warped_target = temporal_warp_core(
                temporal_source_img, temporal_reference_depth, T, K, invK
            )
        
        # Check non-zero regions
        non_zero_mask = (stereo_warped_target != 0) & (temporal_warped_target != 0)
        
        # Compute stereo-temporal difference
        stereo_temporal_diff = self._mask_meansurement(
            stereo_warped_target, temporal_warped_target
        )
        
        # Create occlusion mask
        possible_occlu_mask = torch.zeros((b, 1, h, w), device=target_img.device)
        possible_occlu_mask[stereo_temporal_diff < self.threshold_ratio] = 1
        possible_occlu_mask[torch.logical_not(non_zero_mask)] = 0
        
        return possible_occlu_mask.detach()
 


    def forward(
        self,
        target_img,
        stereo_source_img,
        stereo_reference_disparity,
        stereo_direction,
        temporal_source_img,
        temporal_reference_depth,
        T,
        K,
        invK,
        stereo_warped_target=None,
        temporal_warped_target=None
    ):
        """Forward pass to generate STC mask.
        
        Args:
            See _forward() for argument descriptions.
            
        Returns:
            Tensor: Occlusion mask
        """
        return self._forward(
            target_img,
            stereo_source_img,
            stereo_reference_disparity,
            stereo_direction,
            temporal_source_img,
            temporal_reference_depth,
            T,
            K,
            invK,
            stereo_warped_target,
            temporal_warped_target
        )


@MASKS.register_module()
class STCMaskFast2(nn.Module):
    """Fast stereo-temporal consistency mask generator (version 2).
    
    This version only requires pre-warped images for consistency check.
    
    Args:
        alpha (float): Alpha parameter. Default: 0.2
        num_views (int): Number of views. Default: 1
        num_frames (int): Number of frames. Default: 2
        num_level (int): Number of pyramid levels. Default: 1
        default_view (str): Default view name. Default: "left"
        error_metric (list): List of error metric configurations
        threshold_type (str): Type of threshold. Default: "abs"
        mask_percent (float, optional): Mask percentage
        threshold_ratio (float): Threshold ratio for consistency check. Default: 0.8
        simple_occlu (bool): Use simple occlusion detection. Default: False
        old_api (bool): Use old API. Default: False
    """
    
    def __init__(
        self,
        alpha=0.2,
        num_views=1,
        num_frames=2,
        num_level=1,
        default_view="left",
        error_metric=None,
        threshold_type="abs",
        mask_percent=None,
        threshold_ratio=0.8,
        simple_occlu=False,
        old_api=False
    ):
        super().__init__()
        if error_metric is None:
            error_metric = [dict(type="ZNCCLoss", ratio=1)]
        
        self.alpha = alpha
        self.num_views = num_views
        self.num_frames = num_frames
        self.num_level = num_level
        self.simple_occlu = simple_occlu
        self.old_api = old_api
        self.mask_percent = mask_percent
        self.views = [default_view] if num_views == 1 else ["left", "right"]
        self.error_metric = error_metric
        self.threshold_type = threshold_type
        self.threshold_ratio = threshold_ratio
        self.error_func_list = [build_loss(metric) for metric in error_metric]

    def _mask_meansurement(self, a, b):
        """Measure error between two tensors using multiple metrics.
        
        Args:
            a (Tensor): First tensor
            b (Tensor): Second tensor
            
        Returns:
            Tensor: Mean error across all metrics
        """
        errors = [func(a, b) for func in self.error_func_list]
        return torch.mean(torch.stack(errors), 0)

    def _forward(self, stereo_warped_target=None, temporal_warped_target=None):
        """Internal forward pass for mask generation.
        
        Args:
            stereo_warped_target (Tensor): Pre-computed stereo warped image
            temporal_warped_target (Tensor): Pre-computed temporal warped image
            
        Returns:
            Tensor: Occlusion mask of shape (B, 1, H, W)
        """
        b, c, h, w = stereo_warped_target.shape
        
        # Check non-zero regions
        non_zero_mask = (
            (stereo_warped_target.sum(1, True) != 0) &
            (temporal_warped_target.sum(1, True) != 0)
        )
        
        # Compute stereo-temporal difference
        stereo_temporal_diff = self._mask_meansurement(
            stereo_warped_target, temporal_warped_target
        )
        
        # Create occlusion mask
        possible_occlu_mask = torch.zeros((b, 1, h, w), device=stereo_warped_target.device)
        possible_occlu_mask[stereo_temporal_diff < self.threshold_ratio] = 1
        possible_occlu_mask[torch.logical_not(non_zero_mask)] = 0
        
        return possible_occlu_mask.detach()

    def forward(self, stereo_warped_target=None, temporal_warped_target=None):
        """Forward pass to generate STC masks for multiple levels.
        
        Args:
            stereo_warped_target (Tensor or list): Stereo warped image(s)
            temporal_warped_target (Tensor or list): Temporal warped image(s)
            
        Returns:
            list: List of occlusion masks
        """
        if not isinstance(stereo_warped_target, (list, tuple)):
            stereo_warped_target = [stereo_warped_target]
        if not isinstance(temporal_warped_target, (list, tuple)):
            temporal_warped_target = [temporal_warped_target]
        
        assert len(stereo_warped_target) == len(temporal_warped_target)
        
        occlu_masks = []
        for i in range(len(stereo_warped_target)):
            occlu_masks.append(
                self._forward(stereo_warped_target[i], temporal_warped_target[i])
            )
        
        return occlu_masks

    def loss(self, losses, mask):
        """Apply mask to loss values.
        
        Args:
            losses (dict or list): Loss dictionary or list of loss dictionaries
            mask (list): List of mask tensors
            
        Returns:
            list: Masked loss dictionaries
        """
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        
        for i in range(len(losses)):
            for k in losses[i].keys():
                losses[i][k] = losses[i][k] * mask[i].float()
        
        return losses
        