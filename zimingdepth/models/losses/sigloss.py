'''
Author: Ziming Liu
Date: 2023-02-07 00:26:37
LastEditors: Ziming Liu
LastEditTime: 2023-02-15 22:16:59
Description: ...
Dependent packages: don't need any extral dependency
'''
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..registry import LOSSES
from ...core.visulization import vis_depth_tensor,vis_img_tensor

@LOSSES.register_module()
class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
                #(sum(valid_mask)/sum(torch.ones_like(valid_mask)))
            input = input[valid_mask]
            target = target[valid_mask]
            #print("pred >> ")
            #print(input[:200])
            #print("gt label >> ")
            #print(target[:200])
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        if not isinstance(depth_pred, (list, tuple)):
            depth_pred = [depth_pred]
        losses = dict()
        for i, depth_i in enumerate(depth_pred):
            loss_depth = self.loss_weight * self.sigloss(depth_i, depth_gt)
            #vis_depth_tensor(depth_gt[0].detach(), "/home/ziliu/vis/monodepth6","gtdepth")
            #vis_depth_tensor(depth_i[0].detach(), "/home/ziliu/vis/monodepth6","preddepth")
            losses.update({f"loss_{i}": loss_depth})
        return losses


@LOSSES.register_module()
class SiLogLoss(nn.Module):
    """
    This implementation is from GLPDepth github code.
    """
    def __init__(self, lambd=0.5, weights=[1,], name=None):
        super().__init__()
        self.lambd = lambd
        if name is None:
            name = "0"
        self.name = name
        self.weights = weights

    def forward(self, pred, target):
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        losses = dict()
        for i in range(len(pred)):
            #vis_depth_tensor(target[0].detach(), "/home/ziliu/vis/monodepth6","gtdepth")
            #vis_depth_tensor(pred[i][0].detach(), "/home/ziliu/vis/monodepth6","preddepth")
            valid_mask = (target > 0).detach()
            diff_log = torch.log(target[valid_mask]) - torch.log(pred[i][valid_mask])
            loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                            self.lambd * torch.pow(diff_log.mean(), 2))
            losses.update( {f"loss_{self.name}_siglog_{i}": loss*self.weights[i]})
        return losses