'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-06-24 23:28:29
LastEditors: Ziming Liu
LastEditTime: 2023-06-24 23:32:39
'''
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict

from ..registry import HEADS

@HEADS.register_module()
class PladePoseNet(nn.Module):
   def __init__(self, batchNorm, num_ep):
       super(PladePoseNet, self).__init__()
       self.relu = nn.ReLU()

       # An additional 1x1 conv layer on the logits (not shown in paper). Its contribution should not be much.
       self.convs = OrderedDict()
       self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, 1, 1)
       self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
       self.convs[("pose", 2)] = nn.Conv2d(256, 6, 1)
       self.net = nn.ModuleList(list(self.convs.values()))
       
       for k, v in self.convs.items():
           nn.init.kaiming_normal_(v.weight.data)  # initialize weigths with normal distribution
           v.bias.data.zero_()  # initialize bias as zero

   def forward(self, dlog):   
       out = dlog
       for i in range(3):
           out = self.convs[("pose", i)](out)
           if i != 2:
               out = self.relu(out)

       out = out.mean(3).mean(2)

       out = 0.01 * out.view(-1, 1, 1, 6)

       axisangle = out[..., :3]
       translation = out[..., 3:]

       return axisangle, translation
   
   def loss(self, axisangle, translation, axisangle_gt, translation_gt, weights=None):
       '''
       Description: Supervised pose loss
       Args:: 
       Returns:: 
       '''        
       if weights is None:
           weights = [100, 100, 100, 1, 1, 1]
       assert len(weights) == 6
       weights = torch.tensor(weights).to(axisangle.device).float()
       loss = torch.mean(weights * torch.abs(axisangle - axisangle_gt) ** 2) + torch.mean(weights * torch.abs(translation - translation_gt) ** 2)
       return loss
        

