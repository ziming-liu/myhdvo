'''
Author: Ziming Liu
Date: 2022-07-01 00:39:10
LastEditors: Ziming Liu
LastEditTime: 2023-07-06 18:37:14
Description: ...
Dependent packages: don't need any extral dependency
'''
from abc import ABCMeta, abstractmethod
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from ...core import tensor_img_denorm

from hdvo.models.losses import ZNCC, ZNCCLoss

from ..builder import build_loss

from ..registry import MASKS

from hdvo.core.visulization import *
import os

@MASKS.register_module()
class HomoMaskFast(nn.Module):
    def __init__(self,  avg_kernel=(5,5), 
                   max_kernel=(5,5),  threshold=2, ):
        super().__init__()
        self.avg_kernel = avg_kernel
        self.max_kernel = max_kernel
        self.threshold = threshold
        self.avgpool = torch.nn.AvgPool2d(kernel_size=self.avg_kernel, stride=(1,1), padding= (int((self.avg_kernel[0]-1)/2), int((self.avg_kernel[1]-1)/2))).cuda()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.max_kernel, stride=(1,1), padding= (int((self.max_kernel[0]-1)/2), int((self.max_kernel[1]-1)/2))).cuda()
 

    def _each_homo_mask(self, x):
        b, c, h, w = x.shape
        x_mean = self.avgpool(x)
        x_mean = self.avgpool(x_mean)
        x= torch.abs(x-x_mean)
        x = self.maxpool(x)
        
        mask = torch.ones(x.shape, device=x.device)
        mask[x<(self.threshold)] = 0 # mask homogenous region
        mask_out, max_idx = torch.max(mask,1,True) # return bx1xhxw
        return mask_out


    def forward(self, x):
        '''
        description:  
        return: {*} [[bx1xhxw], ...]  homogeneous masks
        '''     
        if x.shape[1] > 1:
            x = x.mean(1,True)   
        assert x.shape[1] == 1, "use gray scale image"
        return self._each_homo_mask(x).detach()

    
    def loss(self, losses, mask):
        if not isinstance(losses, (list,tuple)):
            losses = [losses]
        if not isinstance(mask, (list,tuple)):
            mask = [mask for _ in range(len(losses))] # if lam is one, extend to multi level 
            
        for i in range(len(losses)):
            for k, v in losses[i].items():
                losses[i][k] = losses[i][k] * mask[i].float()
        return losses
        

        
        