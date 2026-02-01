"""
PoseNet Head Module.

This module implements a pose estimation network head for predicting
camera motion (rotation and translation) between frames.

Reference: https://github.com/nianticlabs/monodepth2

Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-06-24
Last Modified: 2023-10-04
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from ..registry import HEADS

@HEADS.register_module()
class PoseNetHead(nn.Module):
    """Pose estimation network head.
    
    Predicts camera pose (rotation and translation) between consecutive frames.
    
    Args:
        num_ch_enc (list): Number of channels at each encoder level.
        num_input_features (int): Number of input feature maps.
        num_frames_to_predict_for (int, optional): Number of frames to predict
            pose for. Defaults to num_input_features - 1.
        stride (int): Convolution stride. Defaults to 1.
    """
    
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseNetHead, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        """Forward pass for pose prediction.
        
        Args:
            input_features: Input feature maps from encoder.
            
        Returns:
            tuple: (axisangle, translation) where:
                - axisangle: Rotation as axis-angle representation.
                - translation: 3D translation vector.
        """
        out = self.relu(self.convs["squeeze"](input_features[-1]))
        
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

    def loss(self, axisangle, translation, axisangle_gt, translation_gt, weights=None):
        """Compute supervised pose loss.
        
        Args:
            axisangle: Predicted rotation (axis-angle).
            translation: Predicted translation.
            axisangle_gt: Ground truth rotation.
            translation_gt: Ground truth translation.
            weights (list, optional): Loss weights for each dimension.
                Defaults to [100, 100, 100, 1, 1, 1].
                
        Returns:
            Tensor: Weighted pose loss.
        """
        if weights is None:
            weights = [100, 100, 100, 1, 1, 1]
        assert len(weights) == 6
        weights = torch.tensor(weights).to(axisangle.device).float()
        loss = torch.mean(weights * torch.abs(axisangle - axisangle_gt) ** 2) + \
               torch.mean(weights * torch.abs(translation - translation_gt) ** 2)
        return loss