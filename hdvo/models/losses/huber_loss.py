import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES

@LOSSES.register_module()
class HUBERLoss(nn.Module):
    def __init__(self, ratio=0.85, delta=None ):
        super().__init__()
        self.ratio = ratio
        self.delta = delta

    def forward(self, pred, target,):
        if self.delta is not None:
            delta = self.delta
        else:
            input = (pred-target)
            thresh = self.MAD(input.detach())
            delta = float(thresh)
        huberloss = F.huber_loss(pred, target, reduction='none', delta=delta)
        huberloss = huberloss.mean(1, True)
        return self.ratio * huberloss
    
    def MAD(self, input):
        input = input.type(torch.float64)
        if len(input.shape) !=2:
            input = input.reshape([input.shape[0],-1]) # batch x pixels
        num_inliers = 0 
        num_outliers = 0
        # Compute median of absolute deviations (MAD)
        size = input.shape[-1]
        assert size > 1 
        # Compute median
        sorted_input, idx = torch.sort(input, dim=1)
        median = sorted_input[:, size//2] if size%2 else 0.5*(sorted_input[:, size//2-1]+sorted_input[:, size//2])
        #    // For each element, compute absolute deviation (don't care if the order was changed by median)
        adfrommedian = torch.abs(input - median.reshape([-1,1]))
        # // Compute median of absolute deviations
        sorted_adfrommedian, idx = torch.sort(adfrommedian, dim=1)
        ret_mad = sorted_adfrommedian[:, size//2] if size%2 else 0.5*(sorted_adfrommedian[:, size//2-1]+sorted_adfrommedian[:, size//2])
        ret_mad = ret_mad.reshape([-1,1])
        sigma = (ret_mad * 1.4826)
        sigma_minimal = torch.finfo(torch.float64).min
        sigma_maximal = torch.finfo(torch.float64).max
        sigma[sigma < sigma_minimal] = sigma_minimal
        sigma[sigma > sigma_maximal] = sigma_maximal

        thresh = 1.2816 * sigma
        thresh[thresh < 1e-5] = torch.FloatTensor([1e-5]).type_as(input)
        assert thresh.shape[1] == 1 and len(thresh.shape)==2 and thresh.shape[0]==input.shape[0]
        thresh = torch.max(thresh.reshape(-1) )
        return thresh