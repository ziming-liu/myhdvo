'''
Author: Ziming Liu
Date: 2022-06-30 17:19:13
LastEditors: Ziming Liu
LastEditTime: 2023-07-08 03:47:54
Description: get_smooth_loss refers to monodepth2 paper's code
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES


    
@LOSSES.register_module()
class DispSmoothLoss(nn.Module):
    def __init__(self, ratio=1e-3 ):
        super().__init__()
        self.ratio = ratio
    
    def _get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
        grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)
        grad_disp_x =  grad_disp_x.mean()
        grad_disp_y = grad_disp_y.mean()
        return  grad_disp_x + grad_disp_y

    def forward(self, disp, target_img):
        """
        the smooth loss on predicted disparity 
        target_img: the image of time t
        disp: the correponding pred disparity of time t
        """
        mean_disp = disp.mean(2, True).mean(3, True) # b c 1 1  # a trick in direct video undepth
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = self._get_smooth_loss(norm_disp, target_img)
        loss = self.ratio * smooth_loss #/ (2 ** scale)
        return loss

