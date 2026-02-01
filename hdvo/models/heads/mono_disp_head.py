"""
Monocular Disparity Head Module.

This module implements a monocular disparity prediction head with optional
learnable upsampling for depth estimation tasks.

Author: Ziming Liu
Date: 2023-02-08
Last Modified: 2023-08-16
"""

import torch
from torch import nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from mmcv.runner import BaseModule, auto_fp16

from ..builder import build_loss
from ..registry import HEADS
from .base_stereo_head import BaseStereoHead
from ..stereo_predictor.igevstereo_submodules.submodule import BasicConv_IN

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def context_upsample(disp_low, up_weights):
    """Context-aware upsampling for disparity maps.
    
    Args:
        disp_low (Tensor): Low resolution disparity, shape (B, 1, H, W).
        up_weights (Tensor): Upsampling weights, shape (B, 9, 4*H, 4*W).
        
    Returns:
        Tensor: Upsampled disparity map, shape (B, 1, 4*H, 4*W).
    """
    b, c, h, w = disp_low.shape
        
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)
        
    return disp.unsqueeze(1)


@HEADS.register_module()
class MonoDispHead(BaseModule):
    """Monocular disparity prediction head.
    
    This head predicts disparity maps from monocular image features with
    optional learnable upsampling.
    
    Args:
        max_depth (float): Maximum depth value.
        in_channel (int): Number of input channels.
        losses (dict, optional): Loss function configuration.
        latent_channel (int, optional): Number of latent channels. 
            Defaults to in_channel.
        out_channel (int): Number of output channels. Defaults to 1.
        scale_factor (int): Upsampling scale factor. Defaults to 4.
        learn_upsample (bool): Whether to use learnable upsampling. 
            Defaults to False.
    """
    
    def __init__(self, max_depth,  in_channel, losses=None, latent_channel=None, out_channel=1,
                 scale_factor=4, learn_upsample=False ):
        super().__init__()
        self.max_depth = max_depth
        self.scale_factor = scale_factor
        self.learn_upsample = learn_upsample
        if latent_channel is None:
            latent_channel = in_channel
        self.last_layer_depth = nn.Sequential(
                        nn.Conv2d(in_channel, latent_channel, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(latent_channel),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(latent_channel, out_channel, kernel_size=3, stride=1, padding=1),
                        )
        #for m in self.last_layer_depth.modules():
        #    if isinstance(m, nn.Conv2d):
        #        normal_init(m, std=0.001, bias=0)
        if losses is not None:
            self.pred_loss = build_loss(losses)
        

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx0 = nn.Sequential(
                        nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=False),
                        nn.ConvTranspose2d(2*32, 64, kernel_size=4, stride=2, padding=1)
                        )
        
    def forward(self, x ):
        """Forward pass for disparity prediction.
        
        Args:
            x: Input features, can be a single tensor or tuple/list of tensors.
            
        Returns:
            Tensor: Predicted disparity map with sigmoid activation.
        """
        if isinstance(x, (list,tuple)):
            x = x[0]
        if not self.learn_upsample:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", )
        else:
            assert self.scale_factor == 4
            xspx = self.spx0(x)
            spx_pred = self.spx(xspx) # x 4 scale
            spx_pred = F.softmax(spx_pred, 1) # b 9 h w  [0~1]
            x = context_upsample(x, spx_pred.float())

        x1 = self.last_layer_depth(x)
        z = torch.sigmoid(x1)
        
        return z
    
    def loss(self, pred, gt):
        """Compute prediction loss.
        
        Args:
            pred: Predicted disparity.
            gt: Ground truth disparity.
            
        Returns:
            Loss value.
        """
        return self.pred_loss(pred, gt)
    
 
    