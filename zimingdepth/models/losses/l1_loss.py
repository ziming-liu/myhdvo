'''
Author: Ziming Liu
Date: 2022-06-30 17:27:05
LastEditors: Ziming
LastEditTime: 2022-07-02 00:26:22
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES
from abc import ABCMeta, abstractmethod

@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(self, ratio=0.85 ):
        super().__init__()
        self.ratio = ratio
        

    def forward(self, pred, target,):
        l1abs_diff = torch.abs(pred - target) #torch.abs(pred- target)
        l1abs_diff = l1abs_diff.mean(1,True) # b 1 h w
        return self.ratio * l1abs_diff