'''
Author: Ziming Liu
Date: 2022-07-07 23:24:47
LastEditors: Ziming Liu
LastEditTime: 2023-03-09 18:12:30
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn

from hdvo.models.backbones.psmnet_base  import bn_relu_conv3d


from ...registry import COST_AGGREGATORS

@COST_AGGREGATORS.register_module()
class AnyNetAggregator(nn.Module):
    """
    Args:
        in_planes (int): the channels of raw cost volume
        agg_planes (int): the channels of middle 3d convolution layer
        num, (int): the number of middle 3d convolution layer
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): raw cost volume,
                in [BatchSize, in_planes, MaxDisparity, Height, Width] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout
    """

    def __init__(self, in_planes=1, agg_planes=4, num=4, batch_norm=True):
        super(AnyNetAggregator, self).__init__()
        self.in_planes = in_planes
        self.agg_planes = agg_planes
        self.num = num
        self.batch_norm = batch_norm

        self.agg_list =  [bn_relu_conv3d(batch_norm, in_planes, agg_planes, kernel_size=3,
                                    stride=1, padding=1, dilation=1, bias=True)]
        self.agg_list += [bn_relu_conv3d(batch_norm, agg_planes, agg_planes, kernel_size=3,
                                    stride=1, padding=1, dilation=1, bias=True) for _ in range(num)]
        self.agg_list += [bn_relu_conv3d(batch_norm, agg_planes, 1, kernel_size=3,
                                    stride=1, padding=1, dilation=1, bias=True)]
        self.agg = nn.Sequential(*self.agg_list)

    def forward(self, raw_cost):
        # in: [B, in_planes, D, H, W], out: [B, 1, D, H, W]
        cost = self.agg(raw_cost)
        # [B, D, H, W]
        cost = cost.squeeze(dim=1)

        return [cost]


