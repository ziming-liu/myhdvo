'''
Author: Ziming Liu
Date: 2023-02-08 21:59:31
LastEditors: Ziming
LastEditTime: 2023-02-13 00:01:30
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
from torch import nn as nn
from ..builder import build_loss
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from ..registry import HEADS
from .base_stereo_head import BaseStereoHead
from mmcv.runner import BaseModule, auto_fp16

@HEADS.register_module()
class MonoDispHead(BaseModule):
    def __init__(self, max_depth, losses, in_channel, latent_channel=None, out_channel=1 ):
        super().__init__()
        self.max_depth = max_depth
        if latent_channel is None:
            latent_channel = in_channel
        self.last_layer_depth = nn.Sequential(
                        nn.Conv2d(in_channel, latent_channel, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(latent_channel),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(latent_channel, out_channel, kernel_size=3, stride=1, padding=1))
        #for m in self.last_layer_depth.modules():
        #    if isinstance(m, nn.Conv2d):
        #        normal_init(m, std=0.001, bias=0)

        self.pred_loss = build_loss(losses)

    def forward(self, x ):
        x = self.last_layer_depth(x)
        return torch.sigmoid(x) * self.max_depth
    
    def loss(self, pred, gt):
        return self.pred_loss(pred, gt)