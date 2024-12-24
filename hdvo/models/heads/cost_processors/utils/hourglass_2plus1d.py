'''
Author: Ziming Liu
Date: 2022-11-18 17:46:29
LastEditors: Ziming
LastEditTime: 2022-11-18 18:57:18
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from hdvo.models.backbones.psmnet_base  import conv3d_bn, conv3d_bn_relu, conv_bn_relu, deconv3d_bn


class Hourglass2plus1D(nn.Module):
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
        super(Hourglass2plus1D, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = nn.Sequential(conv3d_bn_relu(
            self.batch_norm, in_planes, in_planes * 2,
            kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False
        ))

        self.conv2 = nn.Sequential(
        conv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False
        ),
        conv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,1,3), stride=(1,1,1), padding=(0,0,1), bias=False
        ),
        conv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0), bias=False
        ))

        self.conv3 = nn.Sequential(conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0), bias=False
        ),
        conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,1,3), stride=(1,1,2), padding=(0,0,1), bias=False
        ),
        conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,3,1), stride=(1,2,1), padding=(0,1,0), bias=False
        ))
        self.conv4 = nn.Sequential(conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False
        ),
        conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,1,3), stride=(1,1,1), padding=(0,0,1), bias=False
        ),
        conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0), bias=False
        ))
        self.conv5 = nn.Sequential(deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(3,1,1), padding=(1,0,0), output_padding=(1,0,0), stride=(2,1,1), bias=False
        ),
        deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,1,3), padding=(0,0,1), output_padding=(0,0,1), stride=(1,1,2), bias=False
        ),
        deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=(1,3,1), padding=(0,1,0), output_padding=(0,1,0), stride=(1,2,1), bias=False
        ))
        self.conv6 = nn.Sequential(deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes,
            kernel_size=(3,3,3), padding=(1,1,1), output_padding=(1,1,1), stride=(2,2,2), bias=False
        ),
        )

    def forward(self, x, presqu=None, postsqu=None):
        # in: [B, C, D, H, W], out: [B, 2C, D, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/2, W/2]
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/4, W/4]
        out = self.conv4(out)

        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/2, W/2]
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        out = self.conv6(post)

        return out, pre, post
