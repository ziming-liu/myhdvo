"""
Base Stereo Head Module.

This module provides the abstract base class for all stereo matching heads.
Implementations should define cost building, matching, and disparity prediction.

Reference: https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark

Author: Ziming Liu
Date: 2022-07-07
Last Modified: 2023-07-06
"""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..registry import HEADS

# Concatenate left and right feature to form cost volume
@HEADS.register_module()
class BaseStereoHead(nn.Module):
    """Base class for stereo matching heads.
    
    This abstract base class provides the core interface and common functionality
    for stereo matching neural network heads. Subclasses should implement the
    abstract methods for cost building, matching, and disparity prediction.
    
    Args:
        in_channels (int): Number of input feature channels.
        disp_range (tuple): Disparity range (start, max, dilation).
        alpha (float): Alpha parameter for normalization.
        normalize (bool): Whether to apply normalization.
        losses (dict, optional): Configuration for disparity loss function.
        smooth_loss (dict, optional): Configuration for smoothness loss.
        stereo_feat_cons_losses (dict, optional): Configuration for feature
            consistency loss.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, in_channels, disp_range,  alpha, normalize, losses=None,
                 smooth_loss=None, stereo_feat_cons_losses=None, **kwargs):
        super(BaseStereoHead, self).__init__()
        self.in_channels = in_channels
        self.disp_range = disp_range 
        self.start_disp = disp_range[0]
        self.max_disp = disp_range[1]
        self.end_disp = disp_range[1]-1
        self.dilation = disp_range[2]
        self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation
        # generate disparity sample, in [disp_sample_number,] layout
        self.disp_sample = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        )

        self.alpha = alpha
        self.normalize = normalize

        if losses is not None:
            self.disp_loss_func = build_loss(losses)
            
        if smooth_loss is not None:
            self.smooth_loss_func = build_loss(smooth_loss)
        else:
            self.smooth_loss_func = None
        self.stereo_feat_cons_losses = stereo_feat_cons_losses
        if stereo_feat_cons_losses is not None:
            self.stereo_feat_cons_losses_func = build_loss(stereo_feat_cons_losses)

    @abstractmethod
    def disp_predictor(self, final_costs):
        """Predict disparity from final cost volumes.
        
        Args:
            final_costs: Final processed cost volumes.
            
        Returns:
            Predicted disparity maps.
        """
        pass
    
    @abstractmethod
    def cost_matcher(self, costs):
        """Match and process cost volumes.
        
        Args:
            costs: Raw cost volumes.
            
        Returns:
            Decoded feature representations.
        """
        pass

    @abstractmethod
    def cost_builder(self, stereo_features):
        """Build cost volumes from stereo features.
        
        Args:
            stereo_features: Left and right image features.
            
        Returns:
            Raw cost volumes.
        """
        pass

    def forward(self, stereo_features):
        """Forward pass of the stereo head.
        
        Args:
            stereo_features: Stereo image features.
            
        Returns:
            Predicted disparity maps.
        """
        raw_costs = self.cost_builder(stereo_features)
        decoded_features = self.cost_matcher(raw_costs)
        pred_disps = self.disp_predictor(decoded_features)

        return pred_disps


    def loss(self, pred, gt, **kwargs):
        """Compute disparity loss.
        
        Args:
            pred: Predicted disparity.
            gt: Ground truth disparity.
            **kwargs: Additional arguments.
            
        Returns:
            dict: Dictionary containing loss values.
        """
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses

    def loss_smooth(self, disp, img, **kwargs):
        """Compute smoothness loss.
        
        Args:
            disp: Disparity map.
            img: Input image for edge-aware smoothing.
            **kwargs: Additional arguments.
            
        Returns:
            Smoothness loss value.
        """
        loss = self.smooth_loss_func(disp, img)
        return loss

    def loss_rectify_calib(self, left_feat, right_feat, left_gtdisp=None, right_gtdisp=None, **kwargs):
        """Compute rectification calibration loss.
        
        Args:
            left_feat: Left image features.
            right_feat: Right image features.
            left_gtdisp: Ground truth left disparity.
            right_gtdisp: Ground truth right disparity.
            **kwargs: Additional arguments.
            
        Returns:
            dict: Dictionary containing loss values.
        """
        losses = {}
        losses.update(self.stereo_feat_cons_losses_func(left_feat, right_feat, left_gtdisp, right_gtdisp, **kwargs))
        return losses
