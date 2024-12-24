'''
Author: Ziming Liu
Date: 2023-02-08 21:59:31
LastEditors: Ziming Liu
LastEditTime: 2023-08-16 15:17:00
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
from torch import nn as nn
import torch.nn.functional as F
from ..builder import build_loss
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from ..registry import HEADS
from .base_stereo_head import BaseStereoHead
from mmcv.runner import BaseModule, auto_fp16
from ..stereo_predictor.igevstereo_submodules.submodule import *

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
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape
        
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)
        
    return disp.unsqueeze(1)


@HEADS.register_module()
class MonoDispHead(BaseModule):
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
        z = torch.sigmoid(x1) #* self.max_depth
        
        return z 
        #return torch.relu(x) * self.max_depth
    
    def loss(self, pred, gt):
        return self.pred_loss(pred, gt)
    
 
    