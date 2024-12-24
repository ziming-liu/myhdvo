'''
Author: Ziming Liu
Date: 2022-07-07 23:24:47
LastEditors: Ziming
LastEditTime: 2022-07-14 10:28:29
Description: ...
Dependent packages: don't need any extral dependency
'''
from curses import raw
import torch
import torch.nn as nn

from hdvo.models.backbones.psmnet_base  import conv3d_bn, conv3d_bn_relu, conv_bn_relu, conv_bn
from hdvo.models.heads.cost_processors.utils.hourglass import Hourglass
from mmcv.cnn import ConvModule, constant_init, kaiming_init

from ...registry import COST_AGGREGATORS
from ....core import resize

@COST_AGGREGATORS.register_module()
class Simple2DAggregator(nn.Module):
    """
    Args:
        max_disp (int): max disparity
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): raw cost volume,
                in [BatchSize, Channels, MaxDisparity//4, Height//4, Width//4] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout
    """

    def __init__(self, max_disp, in_planes=64, num_layer=3, 
                stride=1,
                 dilation=1,
                 num_level=3,
                 upsample=False,
                 style='pytorch',
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False, align_corners=False):
        '''
        description:   
        return: {*}
        '''        
        super(Simple2DAggregator, self).__init__()
        self.max_disp = max_disp
        self.align_corners = align_corners
        self.in_planes = in_planes
        self.upsample = upsample
        if upsample:
            self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners)
            self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        conv1_kernel_size = (2, 3, 3)
        conv1_padding = (0, 1, 1)
        self.conv0 =  nn.Sequential(ConvModule(
                        in_planes,
                        (in_planes+max_disp)//2,
                        kernel_size=(1, 3, 3),
                        stride=(1, 1, 1),
                        padding=(0, 1, 1),
                        dilation=(1, 1, 1),
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d'),
                        act_cfg=dict(type='ReLU')), 
            ConvModule(
                (in_planes+max_disp)//2,
                max_disp,
                kernel_size=(2, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                dilation=(1, 1, 1),
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU')), 
        )
        self.conv1 = nn.Sequential(*[
            ConvModule(
            max_disp,
            max_disp,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)  for _ in range(num_layer)
        ])
        self.conv2 = nn.Sequential(*[
            ConvModule(
            max_disp,
            max_disp,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)  for _ in range(num_layer)
        ])
        self.conv3 = nn.Sequential(*[
            ConvModule(
            max_disp,
            max_disp,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)  for _ in range(num_layer)
        ])
        self.cost_bottleneck = ConvModule(
            num_level*max_disp,
            max_disp,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
      

    def forward(self, raw_cost):
        assert len(raw_cost.shape) == 5
        B, C, N, H, W = raw_cost.shape
        cost0 = self.conv0(raw_cost)
        cost0 = cost0.view((B,self.max_disp,H,W))
        cost1 = self.conv1(cost0)
        out1 = cost1
        #if self.upsample:  cost1 = self.upsample1(cost1)
        ident1 = cost1
        cost2 = self.conv2(cost1)
        cost2 = cost2 + ident1
        out2 = cost2
        #if self.upsample:  cost2 = self.upsample2(cost2)
        ident2 = cost2
        cost3 = self.conv3(cost2)
        #cost3 = cost3+ident2
        out3 = cost3 + ident2
        costs = [out1, out2, out3]
        H, W = out3.shape[-2:]
        for i in range(len(costs)):
            costs[i] = resize(
                costs[i],
                size=(4*H, 4*W),
                mode='bilinear',
                align_corners=self.align_corners)

        costs = torch.cat(costs, 1)
        costs = self.cost_bottleneck(costs)
        return [costs]


