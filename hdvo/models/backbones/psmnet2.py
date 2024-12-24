'''
Author: Ziming Liu
Date: 2022-07-07 16:27:15
LastEditors: Ziming
LastEditTime: 2022-09-07 23:52:53
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from torch.utils import checkpoint as cp

from .psmnet_base import conv_bn, conv_bn_relu, BasicBlock
from ...utils import get_root_logger
from ..registry import BACKBONES


@BACKBONES.register_module()
class PSMNet2(nn.Module):
    """
    Backbone proposed in PSMNet2.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms (Tensor): left image feature maps, in [BatchSize, 32, Height//4, Width//4] layout

        r_fms (Tensor): right image feature maps, in [BatchSize, 32, Height//4, Width//4] layout
    """

    def __init__(self, in_planes=3, batch_norm=True, pretrained=None, align_corners=False, with_cp=False):
        super(PSMNet2, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm
        self.pretrained = pretrained
        self.align_corners = align_corners
        self.with_cp = with_cp

        self.firstconv = nn.Sequential(
            conv_bn_relu(batch_norm, self.in_planes, 32, 3, 2, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, 3, 1, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, 3, 1, 1, 1, bias=False),
        )

        # For building Basic Block
        self.in_planes = 32

        self.layer1 = self._make_layer(batch_norm, BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(batch_norm, BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 2, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.lastconv = nn.Sequential(
            conv_bn_relu(batch_norm, 320, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, dilation=1, bias=False)
        )
        self.init_weights()

    def _make_layer(self, batch_norm, block, out_planes, blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = conv_bn(
                batch_norm, self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1
            )

        layers = []
        layers.append(
            block(batch_norm, self.in_planes, out_planes, stride, downsample, padding, dilation)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(batch_norm, self.in_planes, out_planes, 1, None, padding, dilation)
            )

        return nn.Sequential(*layers)

    def _forward(self, x):
        output_2_0 = self.firstconv(x)
        output_2_1 = self.layer1(output_2_0)
        output_4_0 = self.layer2(output_2_1)
        output_4_1 = self.layer3(output_4_0)
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(
            output_branch1, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=self.align_corners
        )

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(
            output_branch2, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=self.align_corners
        )

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(
            output_branch3, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=self.align_corners
        )

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(
            output_branch4, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=self.align_corners
        )

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input
        B, C, H, W = l_img.shape
        stack_input = torch.cat([l_img,r_img], 0)
        if self.with_cp:
            stack_out = cp.checkpoint(self._forward, stack_input)
        else:
            stack_out = self._forward(stack_input) # 12, 32, 80, 256
        feat = stack_out[:B,...] # only left feat
        feat = feat.reshape((B//2,2)+feat.shape[-3:])
        feat = feat.reshape((B//2,)+(-1,)+feat.shape[-2:])
        
        l_fms = stack_out[:B, ...]
        r_fms = stack_out[B:, ...]
        
        return l_fms, r_fms, feat
