"""
Homogeneous region masking module using SAM (Segment Anything Model).

Author: Ziming Liu
Date: 2022-07-01
Last Modified: 2023-11-17
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS

try:
    from ultralytics import SAM
except ImportError:
    warnings.warn("ultralytics not installed. HomoMaskSAM will not be available.")

@HEADS.register_module()
class HomoMaskSAM(nn.Module):
    """Homogeneous region mask generator using SAM.
    
    This module combines traditional homogeneous region detection with SAM segmentation
    to generate more accurate masks for dynamic regions.
    
    Args:
        num_views (int): Number of views. Default: 2
        num_frames (int): Number of frames. Default: 2
        num_level (int): Number of pyramid levels. Default: 3
        loss_func (list): List of loss function types. Default: ["spr", "tpr"]
        default_view (str): Default view name. Default: "left"
        kernel1_size (tuple): First kernel size. Default: (5, 5)
        kernel2_size (tuple): Second kernel size. Default: (5, 5)
        mask_percent (float, optional): Mask percentage threshold
        threshold (float): Threshold for homogeneous region detection. Default: 0.25
        sam_threshold (float): Threshold for SAM mask overlap. Default: 0.8
        old_api (bool): Use old API. Default: False
    """
    
    def __init__(
        self,
        num_views=2,
        num_frames=2,
        num_level=3,
        loss_func=None,
        default_view="left",
        kernel1_size=(5, 5),
        kernel2_size=(5, 5),
        mask_percent=None,
        threshold=0.25,
        sam_threshold=0.8,
        old_api=False
    ):
        super().__init__()
        if loss_func is None:
            loss_func = ["spr", "tpr"]
        
        self.num_views = num_views
        self.num_frames = num_frames
        self.num_level = num_level
        self.default_view = default_view
        self.mask_percent = mask_percent
        self.loss_func = loss_func
        self.old_api = old_api
        self.views = [default_view] if num_views == 1 else ["left", "right"]
        self.kernel1_size = kernel1_size
        self.kernel2_size = kernel2_size
        self.threshold = threshold
        self.sam_threshold = sam_threshold
        
        # Initialize pooling layers
        k1_pad = (int((kernel1_size[0] - 1) / 2), int((kernel1_size[1] - 1) / 2))
        k2_pad = (int((kernel2_size[0] - 1) / 2), int((kernel2_size[1] - 1) / 2))
        
        self.meanpool2d = nn.AvgPool2d(
            kernel_size=kernel1_size, stride=(1, 1), padding=k1_pad
        ).cuda()
        self.pool2d = nn.MaxPool2d(
            kernel_size=kernel2_size, stride=(1, 1), padding=k2_pad
        ).cuda()

        # Initialize SAM model
        self.sam_model = SAM('sam_l.pt')
        self.sam_model.info()

    def _each_homo_mask(self, x):
        """Generate basic homogeneous mask using pooling operations.
        
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Binary mask of shape (B, 1, H, W)
        """
        b, c, h, w = x.shape
        
        # Apply double mean pooling to smooth
        x_mean = self.meanpool2d(x)
        x_mean = self.meanpool2d(x_mean)
        
        # Compute absolute difference and max pool
        x = torch.abs(x - x_mean)
        x = self.pool2d(x)

        # Create mask
        mask = torch.ones(x.shape, device=x.device)
        mask[x < self.threshold] = 0
        
        # Get maximum across channels
        mask_out, _ = torch.max(mask, 1, True)
        return mask_out


    def forward(self, x, frame_names):
        """Forward pass to generate SAM-refined homogeneous masks.
        
        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W)
            frame_names (list): List of frame file paths
            
        Returns:
            Tensor: Refined binary mask of shape (B, 1, H, W)
        """
        assert x.shape[1] == 1, "Input must be grayscale image"
        
        # Get initial homogeneous mask
        lam_mask = self._each_homo_mask(x).detach()
        b, c, h, w = lam_mask.shape
        
        # Run SAM on each frame
        batch_sam_masks = []
        for frame in frame_names:
            results = self.sam_model.predict(frame)[0]
            sam_masks = results.masks.data
            
            # Resize to match target resolution
            sam_masks = F.interpolate(
                sam_masks.unsqueeze(0).float(),
                size=(h, w),
                mode="nearest"
            ).squeeze(0)
            
            batch_sam_masks.append(sam_masks)
        
        batch_sam_masks = torch.stack(batch_sam_masks).to(x.device)  # B N H W
        
        # Refine masks using SAM results
        batch_refined_lam_mask = []
        for batch_idx in range(b):
            reference = lam_mask[batch_idx, 0, ...].bool()
            target_sams = batch_sam_masks[batch_idx, ...]
            N, H, W = target_sams.shape
            refined_lam_mask = torch.zeros((H, W), device=reference.device)
            
            # Check each SAM mask for overlap with homogeneous regions
            for sam_mask_idx in range(N):
                target_sam = target_sams[sam_mask_idx]
                
                # Compute intersection with non-homogeneous regions
                intersection = torch.logical_and(
                    torch.logical_not(reference), target_sam
                )
                
                # If enough overlap, include this SAM mask
                overlap_ratio = (
                    torch.sum(intersection.long()) /
                    torch.sum(target_sam.long())
                )
                if overlap_ratio >= self.sam_threshold:
                    refined_lam_mask = torch.logical_or(refined_lam_mask, target_sam)
            
            # Invert to get mask for non-dynamic regions
            refined_lam_mask = torch.logical_not(refined_lam_mask)
            batch_refined_lam_mask.append(refined_lam_mask)
        
        batch_refined_lam_mask = torch.stack(batch_refined_lam_mask).unsqueeze(1)
        return batch_refined_lam_mask