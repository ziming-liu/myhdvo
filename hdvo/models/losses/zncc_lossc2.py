'''
Author: Ziming Liu
Date: 2022-06-30 11:27:37
LastEditors: Ziming Liu
LastEditTime: 2023-08-18 14:05:40
Description: The zeros normalized cross correlation code. We compute ZNCC on local regions (5x5). 
            partly refer to https://github.com/yuta-hi/pytorch_similarity/
Dependent packages: torch, numpy
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES





class ZNCC(nn.Module):
    """ N-dimensional normalized cross correlation (NCC)

    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                ratio = 0.85,
                kernel_size=(5,5),
                eps=1e-8,
                return_map=True,
                reduction='mean'):

        super(ZNCC, self).__init__()
        self._ratio = ratio
        self._eps = eps
        self._return_map = return_map
        self._reduction = reduction
        self._kernel_size = kernel_size
        with torch.no_grad():
            #conv2d = torch.nn.conv2d(3,3,(3,3),1,1,) 
            kernel1 = np.ones(kernel_size) * (1/(kernel_size[0]*kernel_size[1]))
            kernel1 = torch.FloatTensor(kernel1).expand(1,1,kernel_size[0],kernel_size[1]).cuda() # outch, ch, k, k 
            self.weight = nn.Parameter(data=kernel1, requires_grad=False).cuda()
            self.pad_size = (int((kernel_size[0]-1)/2), int((kernel_size[1]-1)/2))
            #self.conv_mean = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, stride=(1,1), padding=pad_size,  )
            #self.conv_mean.weight.data = self.weight #TODO: use F.conv2d to compute mean intensity

        
        #self.freeze(self.conv_mean)


    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
                
    def zeros_normalized_cross_correlation(self, x, y, return_map, reduction='mean', eps=1e-8):
        """ zeros normalized cross correlation (ZNCC)

        Args:
            x (~torch.Tensor): Input tensor.
            y (~torch.Tensor): Input tensor.
            return_map (bool): If True, also return the correlation map.
            kernel_size (tuple(,)): give a kernel size to compute local region mean intensity.  
            num_pixel of the patch image is kernelsize[0]*kernelsize[1]
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.

        Returns:
            ~torch.Tensor: Output scalar
            ~torch.Tensor: Output tensor
        refer to https://github.com/yuta-hi/pytorch_similarity/
        """
         
        
        #print(self.conv_mean.weight.data)

        shape = x.shape
        b,c,h,w = shape
        b = shape[0]
        #""" 
        #kernel_size = (101, 101) #(h-1,w-1)
        #with torch.no_grad():
        #x_mean = torch.zeros(x.shape).type_as(x)
        xbc = x.reshape(b*c,1,h,w)
        #x_mean = self.conv_mean(xbc)
        x_mean = F.conv2d(input=xbc,
                            weight=self.weight,
                            stride=(1,1),
                            padding=self.pad_size,
                             )
        x_mean = x_mean.reshape(b,c,h,w)
        #print("x\n",x, "\n xmean3x3\n",x_mean)
        #x_mean[:,0:1,:,:] = self.conv_mean(x[:,0:1,:,:]) #F.conv2d(x[:,0:1,:,:], weight, stride=(1,1), padding=pad_size, padding_mode='reflect')
        #x_mean[:,1:2,:,:] = self.conv_mean(x[:,1:2,:,:]) # F.conv2d(x[:,1:2,:,:], weight, stride=(1,1), padding=pad_size, padding_mode='reflect')
        #x_mean[:,2:3,:,:] = self.conv_mean(x[:,2:3,:,:]) # F.conv2d(x[:,2:3,:,:], weight, stride=(1,1), padding=pad_size, padding_mode='reflect')

        
        ##print("x\n",x, "\n xmean3x3\n",x_mean) 
        #y_mean = torch.zeros(y.shape).type_as(y)
        ybc = y.reshape(b*c,1,h,w)
        #y_mean = self.conv_mean(ybc)
        y_mean = F.conv2d(input=ybc,
                            weight=self.weight,
                            stride=(1,1),
                            padding=self.pad_size,
                             )
        #y_mean[:,0:1,:,:] = self.conv_mean(y[:,0:1,:,:]) # F.conv2d(y[:,0:1,:,:], weight, stride=(1,1), padding=pad_size, padding_mode='reflect')
        #y_mean[:,1:2,:,:] = self.conv_mean(y[:,1:2,:,:]) # F.conv2d(y[:,1:2,:,:], weight, stride=(1,1), padding=pad_size, padding_mode='reflect')
        #y_mean[:,2:3,:,:] = self.conv_mean(y[:,2:3,:,:]) # F.conv2d(y[:,2:3,:,:], weight, stride=(1,1), padding=pad_size, padding_mode='reflect')
        y_mean = y_mean.reshape(b,c,h,w)
        x = x-x_mean # xpatch_min_avg
        y = y-y_mean # ypatch_min_avg
        
        #"""
        # reshape
        #x = x.view(b, -1)
        #y = y.view(b, -1)
        """ 
        # mean
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)
        
        # deviation
        x = x - x_mean
        y = y - y_mean
        """
        mul_xy = x*y #torch.mul(x,y)
        mul_xx = x*x #torch.mul(x,x)
        mul_yy = y*y #torch.mul(y,y)

        #kernel2 = np.ones(kernel_size)
        #kernel2 = torch.FloatTensor(kernel2).expand(1,1,kernel_size[0],kernel_size[1]) # outch, ch, k, k 
        #weight2 = nn.Parameter(data=kernel2, requires_grad=False).type_as(x)
        #conv_sum = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, stride=(1,1), padding=pad_size,  ).to(x.device)
        #conv_sum.weight.data = weight2
        #avg_xy_patch_sum = torch.zeros(mul_xy.shape).type_as(mul_xy)
        mul_xybc = mul_xy.reshape(b*c,1,h,w)
        #print("mul_xybc\n",mul_xybc)
        #avg_xy_patch_sum = self.conv_mean(mul_xybc)
        avg_xy_patch_sum = F.conv2d(input=mul_xybc,
                            weight=self.weight,
                            stride=(1,1),
                            padding=self.pad_size,
                             )
        avg_xy_patch_sum = avg_xy_patch_sum.reshape(b,c,h,w)
        #avg_xy_patch_sum[:,0:1,:,:] = self.conv_mean(mul_xy[:,0:1, :, :])
        #avg_xy_patch_sum[:,1:2,:,:] = self.conv_mean(mul_xy[:,1:2, :, :])
        #avg_xy_patch_sum[:,2:3,:,:] = self.conv_mean(mul_xy[:,2:3, :, :])
        #avg_xy_patch_sum = (1/(kernel_size[0]*kernel_size[1])) * xy_patch_sum
        #print("avg_xy_patch_sum\n",avg_xy_patch_sum)
        #avg_xx_patch_sum = torch.zeros(mul_xx.shape).type_as(mul_xx)
        mul_xxbc = mul_xx.reshape(b*c,1,h,w)
        #avg_xx_patch_sum = self.conv_mean(mul_xxbc)
        avg_xx_patch_sum = F.conv2d(input=mul_xxbc,
                            weight=self.weight,
                            stride=(1,1),
                            padding=self.pad_size,
                             )
        avg_xx_patch_sum = avg_xx_patch_sum.reshape(b,c,h,w)
        #avg_xx_patch_sum[:,0:1,:,:] = self.conv_mean(mul_xx[:,0:1, :, :])
        #avg_xx_patch_sum[:,1:2,:,:] = self.conv_mean(mul_xx[:,1:2, :, :])
        #avg_xx_patch_sum[:,2:3,:,:] = self.conv_mean(mul_xx[:,2:3, :, :])

        #avg_yy_patch_sum = torch.zeros(mul_yy.shape).type_as(mul_yy)
        mul_yybc = mul_yy.reshape(b*c,1,h,w)
        #avg_yy_patch_sum = self.conv_mean(mul_yybc)
        avg_yy_patch_sum = F.conv2d(input=mul_yybc,
                            weight=self.weight,
                            stride=(1,1),
                            padding=self.pad_size,
                             )
        avg_yy_patch_sum = avg_yy_patch_sum.reshape(b,c,h,w)
        #avg_yy_patch_sum[:,0:1,:,:] = self.conv_mean(mul_yy[:,0:1, :, :])
        #avg_yy_patch_sum[:,1:2,:,:] = self.conv_mean(mul_yy[:,1:2, :, :])
        #avg_yy_patch_sum[:,2:3,:,:] = self.conv_mean(mul_yy[:,2:3, :, :])
        #avg_yy_patch_sum = (1/(kernel_size[0]*kernel_size[1])) * yy_patch_sum
        term1 = avg_xy_patch_sum
        term2 = torch.sqrt(avg_xx_patch_sum)*torch.sqrt(avg_yy_patch_sum)
        #eps2 = torch.zeros(term2.shape).to(x.device)
        #eps2[term2==0] = eps
        #ncc_map = torch.div(term1 + eps2, 
        #                term2 + eps2)  # bxcxhxw
        
        ncc_map = term1 / (term2 + eps)
        
        ncc_map = torch.mean(ncc_map, 1, True) # bx1xhxw average RGB channel's results
        ncc = torch.mean(ncc_map.reshape(b,-1), 1, False) # batchsize vector
        

        if not return_map:
            return ncc

        return ncc_map

    def _forward(self, x, y):
        #with torch.no_grad():
        ncc_map = self.zeros_normalized_cross_correlation(x, y,
                                self._return_map, self._reduction, self._eps)

        return ncc_map
    
    def forward(self, x, y):
    
        return self._forward(x, y)

@LOSSES.register_module()
class ZNCCLoss(ZNCC):
    """ N-dimensional normalized cross correlation loss (NCC-loss)

    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self, ratio=0.85, kernel_size=(5, 5), eps=1e-8, return_map=True, reduction='mean'):
        super().__init__(ratio, kernel_size, eps, return_map, reduction)
        
    def forward(self, x, y):
        gc = self._forward(x, y)

        if not self._return_map:
            gc = self._ratio * (1.0 - gc)
            return gc

        return self._ratio * torch.clamp((1.0-gc)/2, 0, 1)