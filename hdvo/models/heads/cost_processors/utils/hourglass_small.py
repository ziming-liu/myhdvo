'''
Author: Ziming Liu
Date: 2023-02-15 14:00:27
LastEditors: Ziming Liu
LastEditTime: 2023-02-15 16:27:11
Team: ACENTAURI team, INRIA
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from hdvo.models.backbones.psmnet_base  import conv3d_bn, conv3d_bn_relu, conv3d_bn_relu, deconv3d_bn

from ....models.backbones.vit import TransformerEncoderLayer
from mmcv.cnn import ConvModule, build_norm_layer
from typing import Callable, Optional, Sequence



class HourglassFPNSmall(nn.Module):
    """
    An implementation of hourglass module proposed in PSMNet.
    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer,
            default True
    Inputs:
        x, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        presqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        postsqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
    Outputs:
        out, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        pre, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        post, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout

    """
    def __init__(self, in_planes, batch_norm=True):
        super(HourglassFPNSmall, self).__init__()
        self.batch_norm = batch_norm

    
        self.conv1_0 = self.make_vit_layer(in_planes, in_planes * 2, 4)
        self.conv1_1 = nn.Conv1d(in_planes, in_planes, 2, 2)
        self.conv1_2 = conv3d_bn_relu(batch_norm, in_planes, in_planes * 2, (1,3,3), (1,2,2), (0,1,1), bias=False)
        

        self.conv2_0 = self.make_vit_layer(in_planes * 2, in_planes * 2, 4)
        self.conv2_1 = conv3d_bn_relu(batch_norm, in_planes * 2, in_planes * 2, (1,3,3), 1, (0,1,1), bias=False)

 
        self.conv3_0 = self.make_vit_layer(in_planes * 2, in_planes * 2, 4)
        self.conv3_1 = nn.Conv1d(in_planes*2, in_planes*2, 2, 2)
        self.conv3_2 = conv3d_bn_relu(batch_norm, in_planes * 2, in_planes * 2, (1,3,3), (1,2,2), (0,1,1), bias=False)
        
        self.conv4_0 = self.make_vit_layer(in_planes * 2, in_planes * 2, 4)
        self.conv4_1 = conv3d_bn_relu(batch_norm, in_planes * 2, in_planes * 2,  (1,3,3), 1, (0,1,1), bias=False)
        
        self.conv5_0 = self.make_vit_layer(in_planes * 2, in_planes * 2, 4)
        self.conv5_1 = conv3d_bn_relu(batch_norm, in_planes * 2, in_planes*2,  (1,3,3), 1, (0,1,1), bias=False)

        self.conv6_0 = self.make_vit_layer(in_planes * 2, in_planes * 2, 4)
        self.conv6_1 = conv3d_bn_relu(batch_norm, in_planes * 2, in_planes,  (1,3,3), 1, (0,1,1), bias=False)
        

    def forward(self, x, presqu=None, postsqu=None):
        B, C, D, H, W = x.shape
        # in: [B, C, D, H, W], out: [B, 2C, D/2, H/2, W/2]
        x = x.permute(0,3,4,2,1).reshape(-1,D,C)
        x = self.conv1_0(x)
        x = x.transpose(2,1)
        x = self.conv1_1(x) # D->D/2
        x = x.transpose(2,1)
        x = x.reshape(B,H,W,D//2,C).permute(0,4,3,1,2)
        out = self.conv1_2(x)

        # in: [B, 2C, D//2, H/2, W/2], out: [B, 2C, D//2, H/2, W/2]
        out = out.permute(0,3,4,2,1).reshape(-1,D//2,2*C)
        out = self.conv2_0(out)
        out = out.reshape(B,H//2,W//2,D//2,2*C).permute(0,4,3,1,2)
        pre = self.conv2_1(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=False)
        else:
            pre = F.relu(pre, inplace=False)

        # in: [B, 2C, D/2, H/2, W/2], out: [B, 2C, D/4, H/4, W/4]
        x = pre.permute(0,3,4,2,1).reshape(-1,D//2,2*C)
        x = self.conv3_0(x)
        x = x.transpose(2,1)
        x = self.conv3_1(x) # D//2-> D/4
        x = x.transpose(2,1)
        x = x.reshape(B,H//2,W//2,D//4,2*C).permute(0,4,3,1,2)
        out = self.conv3_2(x)
        # in: [B, 2C, D/4, H/4, W/4], out: [B, 2C, D/4, H/4, W/4]
        out = out.permute(0,3,4,2,1).reshape(-1,D//4,2*C)
        out = self.conv4_0(out)
        out = out.reshape(B,H//4,W//4,D//4,2*C).permute(0,4,3,1,2)
        out = self.conv4_1(out)
        # in: [B, 2C, D/4, H/4, W/4], out: [B, 2C, D/2, H/2, W/2]
        out_up1 = F.interpolate(out, scale_factor=2,mode="trilinear",align_corners=False)
        if presqu is not None:
            post = F.relu(out_up1 + presqu, inplace=False)
        else:
            post = F.relu(out_up1 + pre, inplace=False)
        x = post.permute(0,3,4,2,1).reshape(-1,D//2,2*C)
        x = self.conv5_0(x)
        x = x.reshape(B,H//2,W//2,D//2,2*C).permute(0,4,3,1,2)
        x = self.conv5_1(x)
        # in: [B, 2C, D/2, H/2, W/2], out: [B, C, D, H, W]
        out_up2 = F.interpolate(x, scale_factor=2,mode="trilinear",align_corners=False)
        out_up2 = out_up2.permute(0,3,4,2,1).reshape(-1,D,2*C)
        out_up2 = self.conv6_0(out_up2)
        out_up2 = out_up2.reshape(B,H,W,D,2*C).permute(0,4,3,1,2)
        output = self.conv6_1(out_up2)

        return output, pre, post


    def make_vit_layer(self, 
            transformer_dim,
            ffn_dim,
            num_heads: int = 4,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            transformer_norm_cfg: Callable = dict(type='LN')):

        return nn.Sequential(*[
            TransformerEncoderLayer(
                embed_dims=transformer_dim,
                num_heads=num_heads,
                feedforward_channels=ffn_dim,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=True,
                act_cfg=dict(type='Swish'),
                norm_cfg=transformer_norm_cfg),
            build_norm_layer(transformer_norm_cfg, transformer_dim)[1],
        ])
