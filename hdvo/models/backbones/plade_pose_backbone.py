# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict

from ..registry import BACKBONES



def conv_elu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
   if batchNorm:
       return nn.Sequential(
           nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                     bias=False),
           nn.BatchNorm2d(out_planes),
           nn.ELU(inplace=True)
       )
   else:
       return nn.Sequential(
           nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                     bias=True),
           nn.ELU(inplace=True)
       )


class deconv(nn.Module):
   def __init__(self, in_planes, out_planes):
       super(deconv, self).__init__()
       self.elu = nn.ELU(inplace=True)
       self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

   def forward(self, x, ref):
       x = F.interpolate(x, size=(ref.size(2), ref.size(3)), mode='nearest')
       x = self.elu(self.conv1(x))
       return x


def conv_gep(in_planes, int_planes, out_planes):
   return nn.Sequential(
       nn.Conv2d(in_planes, int_planes, kernel_size=1, stride=1, padding=0, bias=True),
       nn.ELU(inplace=True),
       nn.Conv2d(int_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
   )


class residual_block(nn.Module):
   def __init__(self, in_planes, kernel_size=3):
       super(residual_block, self).__init__()
       self.elu = nn.ELU(inplace=True)
       self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
       self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

   def forward(self, x):
       x = self.elu(self.conv2(self.elu(self.conv1(x))) + x)
       return x

@BACKBONES.register_module()
class PladePoseBackbone(nn.Module):
   def __init__(self, batchNorm=True, no_in=3, no_ep=8):
       super(PladePoseBackbone, self).__init__()
       self.batchNorm = batchNorm

       # Encoder
       # Encode pixel position from 2 to no_ep
       self.conv_ep1 = conv_gep(2, 16, no_ep)
       self.conv_ep2 = conv_gep(2, 16, no_ep)
       self.conv_ep3 = conv_gep(2, 16, no_ep)
       self.conv_ep4 = conv_gep(2, 16, no_ep)
       self.conv_ep5 = conv_gep(2, 16, no_ep)
       self.conv_ep6 = conv_gep(2, 16, no_ep)

       # Two input layers at full and half resolution
       self.conv0 = conv_elu(self.batchNorm, no_in, 64, kernel_size=3)
       self.conv0_1 = residual_block(64)
       self.conv0l = conv_elu(self.batchNorm, no_in, 64, kernel_size=3)
       self.conv0l_1 = residual_block(64)

       # Strided convs of encoder
       self.conv1 = conv_elu(self.batchNorm, 64 + no_ep, 128, pad=1, stride=2)
       self.conv1_1 = residual_block(128)
       self.conv2 = conv_elu(self.batchNorm, 128 + 64 + no_ep, 256, pad=1, stride=2)
       self.conv2_1 = residual_block(256)
       self.conv3 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
       self.conv3_1 = residual_block(256)
       self.conv4 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
       self.conv4_1 = residual_block(256)
       self.conv5 = conv_elu(self.batchNorm, 256 + no_ep, 256, pad=1, stride=2)
       self.conv5_1 = residual_block(256)
       self.conv6 = conv_elu(self.batchNorm, 256 * 2 + no_ep, 256, pad=1, stride=2)
       self.conv6_1 = residual_block(256)
       self.elu = nn.ELU(inplace=True)

       # Initialize conv layers
       for m in self.modules():
           if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
               nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
               if m.bias is not None:
                   m.bias.data.zero_()  # initialize bias as zero
           elif isinstance(m, nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()

   def forward(self, x, y, grid):
       _, _, H, W = x.shape

       ######################################### Left Encoder section##################################################
       # Early feature extraction at target resolution
       out_conv0 = self.conv0_1(self.conv0(x))

       # One strided conv encoder stage
       out_conv1 = self.conv1_1(self.conv1(torch.cat((out_conv0, self.conv_ep1(grid)), 1)))

       # Early geature extraction at half resolution
       out_conv0lr = self.conv0l_1(self.conv0l(
           F.interpolate(x, size=(out_conv1.shape[2], out_conv1.shape[3]), mode='bilinear', align_corners=True)))

       # Deep feature extraction
       dgrid = F.interpolate(grid, size=(out_conv1.shape[2], out_conv1.shape[3]), align_corners=True, mode='bilinear')
       out_conv2 = self.conv2_1(self.conv2(torch.cat((out_conv1, out_conv0lr, self.conv_ep2(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv2.shape[2], out_conv2.shape[3]), align_corners=True, mode='bilinear')
       out_conv3 = self.conv3_1(self.conv3(torch.cat((out_conv2, self.conv_ep3(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv3.shape[2], out_conv3.shape[3]), align_corners=True, mode='bilinear')
       out_conv4 = self.conv4_1(self.conv4(torch.cat((out_conv3, self.conv_ep4(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv4.shape[2], out_conv4.shape[3]), align_corners=True, mode='bilinear')
       out_conv5 = self.conv5_1(self.conv5(torch.cat((out_conv4, self.conv_ep5(dgrid)), 1)))

       ######################################### Right Encoder section##################################################
       # Early feature extraction at target resolution
       out_conv0r = self.conv0_1(self.conv0(y))

       # One strided conv encoder stage
       out_conv1r = self.conv1_1(self.conv1(torch.cat((out_conv0r, self.conv_ep1(grid)), 1)))

       # Early geature extraction at half resolution
       out_conv0lrr = self.conv0l_1(self.conv0l(
           F.interpolate(y, size=(out_conv1r.shape[2], out_conv1r.shape[3]), mode='bilinear', align_corners=True)))

       # Deep feature extraction
       dgrid = F.interpolate(grid, size=(out_conv1r.shape[2], out_conv1r.shape[3]), align_corners=True, mode='bilinear')
       out_conv2r = self.conv2_1(self.conv2(torch.cat((out_conv1r, out_conv0lrr, self.conv_ep2(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv2r.shape[2], out_conv2r.shape[3]), align_corners=True, mode='bilinear')
       out_conv3r = self.conv3_1(self.conv3(torch.cat((out_conv2r, self.conv_ep3(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv3r.shape[2], out_conv3r.shape[3]), align_corners=True, mode='bilinear')
       out_conv4r = self.conv4_1(self.conv4(torch.cat((out_conv3r, self.conv_ep4(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv4r.shape[2], out_conv4r.shape[3]), align_corners=True, mode='bilinear')
       out_conv5r = self.conv5_1(self.conv5(torch.cat((out_conv4r, self.conv_ep5(dgrid)), 1)))

       dgrid = F.interpolate(grid, size=(out_conv5.shape[2], out_conv5.shape[3]), align_corners=True, mode='bilinear')
       out_conv6 = self.conv6_1(self.conv6(torch.cat((out_conv5, out_conv5r, self.conv_ep6(dgrid)), 1)))

       return out_conv6

