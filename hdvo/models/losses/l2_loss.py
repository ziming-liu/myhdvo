'''
Author: Ziming Liu
Date: 2022-06-30 17:27:05
LastEditors: Ziming Liu
LastEditTime: 2023-10-07 20:26:46
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES

@LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self, ratio=0.85 ):
        super().__init__()
        self.ratio = 0.85

    def forward(self, pred, target,):
        l2abs_diff = torch.abs(pred- target)**2
        l2abs_diff = l2abs_diff.mean(1, True)
        return self.ratio * l2abs_diff