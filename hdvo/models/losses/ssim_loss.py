'''
Author: Ziming Liu
Date: 2022-06-30 17:15:48
LastEditors: Ziming
LastEditTime: 2022-07-02 00:29:38
Description: SSIM code refers to the code of Monodepth2 paper.
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES

@LOSSES.register_module()
class SSIMLoss(nn.Module):
    def __init__(self, ratio=0.15):
        super(SSIMLoss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.ratio = ratio

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim_map = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        ssim_map = ssim_map.mean(1,True) # b 1 h w
        return self.ratio * ssim_map