'''
Author: Ziming Liu
Date: 2022-10-07 15:42:51
LastEditors: Ziming Liu
LastEditTime: 2023-07-17 00:03:56
Description: L1 loss for supervised disparity regression loss. Refer to github DenseMatching repo
Dependent packages: torch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES
from abc import ABCMeta, abstractmethod
import warnings
from ...core.visulization import vis_depth_tensor,vis_img_tensor


@LOSSES.register_module()
class DispL1Loss(object):
    """
    Args:
        max_disp (int): the max of Disparity. default is 192
        start_disp (int): the start searching disparity index, usually be 0
        weights (list of float or None): weight for each scale of estCost.
        sparse (bool): whether the ground-truth disparity is sparse,
            for example, KITTI is sparse, but SceneFlow is not, default is False.
    Inputs:
        estDisp (Tensor or list of Tensor): the estimated disparity map,
            in [BatchSize, 1, Height, Width] layout.
        gtDisp (Tensor): the ground truth disparity map,
            in [BatchSize, 1, Height, Width] layout.
    Outputs:
        loss (dict), the loss of each level
    """

    def __init__(self, max_disp, start_disp=0, weight=1, weights=None, disp_reweight=False,
             sparse=False, set_range=True, random_mask=False, random_mask_ratio=0.3,
             name="disp_l1_loss_lvl"):
        self.max_disp = max_disp
        self.weights = weights
        self.name = name
        self.weight = weight # the loss weight relative to other losses
        self.start_disp = start_disp
        self.sparse = sparse
        self.set_range = set_range
        self.random_mask = random_mask
        self.random_mask_ratio=random_mask_ratio
        self.disp_reweight = disp_reweight
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estDisp, gtDisp):
        N, C, H, W = estDisp.shape
        scaled_gtDisp = gtDisp
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp / scale
            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))


        if self.random_mask:
            #print(estDisp.reshape(N, C, H*W).permute(0,2,1).shape)
            estDisp = estDisp.reshape(N, C, H*W).permute(0,2,1) # B H*W 
            scaled_gtDisp = scaled_gtDisp.reshape(N, C, H*W).permute(0,2,1)
            estDisp, scaled_gtDisp = self.random_masking(estDisp, scaled_gtDisp, self.random_mask_ratio)
            #print(estDisp.shape)
        #print(max(scaled_gtDisp.reshape(-1)))
        # mask for valid disparity
        # (start disparity, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        if self.disp_reweight:
            disp_reweight_values = torch.exp(scaled_gtDisp/(self.max_disp//4))
        mask = torch.ones_like(estDisp)
        if self.set_range:
            mask = (scaled_gtDisp > self.start_disp) & (scaled_gtDisp < (self.max_disp / scale))
            assert mask.sum() >= 1.0, 'L1 loss: there is no point\'s disparity is in ({},{})! \n >> \n {}'.format(self.start_disp,
                                                                                        self.max_disp / scale, scaled_gtDisp)
            loss = F.smooth_l1_loss(estDisp[mask], scaled_gtDisp[mask], reduction='none')
            #loss = torch.abs(estDisp[mask]-scaled_gtDisp[mask]).mean()
        else:
            loss = F.smooth_l1_loss(estDisp, scaled_gtDisp, reduction='none')
            #loss = (torch.abs(estDisp - scaled_gtDisp) * mask.float()).mean()
        if self.disp_reweight:
            #print(loss.shape)
            #print("pre loss ", loss.mean())
            #print(disp_reweight_values[mask])
            #print(scaled_gtDisp[mask])
            loss = loss * disp_reweight_values[mask].reshape(-1)
            #print("after loss ", loss.mean())
        return loss * self.weight

        # l1 loss
        #print("estimated ", estDisp[mask])
        #print("gt ", scaled_gtDisp[mask])
        

        #return loss * self.weight

    def __call__(self, estDisp, gtDisp, weights=None, name=None):
        if name is not None:
            self.name = name
        if not isinstance(estDisp, (list, tuple)):
            estDisp = [estDisp]
        #vis_depth_tensor(gtDisp, "/home/ziliu/vis/monodepth3","gtdepth")
        #vis_depth_tensor(estDisp[0], "/home/ziliu/vis/monodepth3","preddepth")
        if self.weights is None:
            self.weights = [1.0] * len(estDisp)
        if weights is not None:
            self.weights = weights
            
        # compute loss for per level
        loss_all_level = []
        for est_disp_per_lvl in estDisp:
            loss_all_level.append(
                self.loss_per_level(est_disp_per_lvl, gtDisp)
            )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = f"{self.name}_{i}"#.format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str


    def random_masking(self, x, xgt, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence, L=H*W, D=1
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        xgt_masked = torch.gather(xgt, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        #mask = torch.ones([N, L], device=x.device)
        #mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        #mask = torch.gather(mask, dim=1, index=ids_restore).long()

        return x_masked, xgt_masked
        #return x_masked, mask, ids_restore

"""
# refer to https://github.com/facebookresearch/mae/blob/main/models_mae.py
 def random_masking(self, x, mask_ratio):
        ""
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        ""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

"""