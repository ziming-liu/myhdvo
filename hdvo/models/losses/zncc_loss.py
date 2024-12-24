'''
Author: Ziming Liu
Date: 2022-06-30 11:27:37
LastEditors: Ziming Liu
LastEditTime: 2023-08-18 13:53:46
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

            self.avgx = nn.AvgPool2d(kernel_size, stride=(1,1), padding=self.pad_size)
            self.avgy = nn.AvgPool2d(kernel_size, stride=(1,1), padding=self.pad_size)
            self.avgxy = nn.AvgPool2d(kernel_size, stride=(1,1), padding=self.pad_size)
            self.avgxx = nn.AvgPool2d(kernel_size, stride=(1,1), padding=self.pad_size)
            self.avgyy = nn.AvgPool2d(kernel_size, stride=(1,1), padding=self.pad_size)
            
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

        x_mean = self.avgx(x)
        y_mean = self.avgy(y)

        x_unfold = F.unfold(x, kernel_size=self._kernel_size, stride=(1,1), padding=self.pad_size)
        y_unfold = F.unfold(y, kernel_size=self._kernel_size, stride=(1,1), padding=self.pad_size)
        x_unfold = x_unfold.reshape((b,c) + self._kernel_size + (h,w))
        y_unfold = y_unfold.reshape((b,c) + self._kernel_size + (h,w))

        x = torch.abs(x_unfold -x_mean.reshape((b,c,1,1,h,w)) ) # xpatch_min_avg
        y = torch.abs(y_unfold -y_mean.reshape((b,c,1,1,h,w)) ) # ypatch_min_avg

        mul_xy = x*y #torch.mul(x,y)
        mul_xx = x*x #torch.mul(x,x)
        mul_yy = y*y #torch.mul(y,y)
        
        ncc_map = (mul_xy+1e-4) / (torch.sqrt(mul_xx+1e-4)*torch.sqrt(mul_yy+1e-4) )

        ncc_map = ncc_map.reshape((b,c, self._kernel_size[0]**2, h,w)).mean(2,False)
        ncc_map = torch.clamp(ncc_map, min=0, max=1)
        #print(ncc_map)
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
        gc = self._forward(x, y.detach())

        if not self._return_map:
            gc = self._ratio * (1.0 - gc)
            return gc
        #print(torch.clamp((1.0-gc), 0, 2)[0,:,100])
        return self._ratio * torch.clamp((1.0-gc), 0, 2)