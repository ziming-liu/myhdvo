'''
Author: Ziming Liu
Date: 2022-07-07 23:33:54
LastEditors: Ziming Liu
LastEditTime: 2023-03-09 18:13:11
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from ...registry import DISP_PREDICTORS

@DISP_PREDICTORS.register_module()
class CascadeSoftArgmin(nn.Module):
    """
    An implementation of cascade soft argmin.
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (optional, int): the step between near disparity index
        alpha (float or int): a factor will times with cost_volume
            details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        normalize (bool): whether apply softmax on cost_volume, default True

    Inputs:
        cost_volume (Tensor): the matching cost after regularization,
            in [BatchSize, disp_sample_number, Height, Width] layout
        disp_sample (optional, Tensor): the estimated disparity samples,
            in [BatchSize, disp_sample_number, Height, Width] layout

    Returns:
        disp_map (Tensor): a disparity map regressed from cost volume,
            in [BatchSize, 1, Height, Width] layout
    """

    def __init__(self, max_disp=192, start_disp=0, dilation=1, alpha=[1., 1., 1.0,], normalize=True, conv_cfg=None, 
                act_cfg=None,
                norm_cfg=None,):
        super(CascadeSoftArgmin, self).__init__()
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + max_disp - 1
        self.disp_sample_number = (max_disp + dilation - 1) // dilation

        self.alpha = alpha
        assert len(self.alpha) == 3, "cascade soft argmin uses 3 alphas"
        self.normalize = normalize

        self.stage1 = ConvModule(
                self.disp_sample_number, 
                self.disp_sample_number, 
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        self.stage2 = ConvModule(
                self.disp_sample_number, 
                self.disp_sample_number, 
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        self.stage3 = ConvModule(
                self.disp_sample_number, 
                self.disp_sample_number, 
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        # generate disparity sample, in [disp_sample_number,] layout
        self.disp_sample = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        )
        assert self.disp_sample_number%2==0
        self.disp_sample_pre = torch.linspace(
            -self.end_disp//2, self.start_disp-1, self.disp_sample_number//2
        )
        self.disp_sample_behind = torch.linspace(
            self.start_disp, self.end_disp//2-1, self.disp_sample_number//2
        )


    def forward(self, cost_volume, disp_sample=None):
        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        if cost_volume.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(cost_volume.dim()))

        ## STAGE 1 
        cost_volume1 = self.stage1(cost_volume)
        # scale cost volume with alpha
        cost_volume1 = cost_volume1 * self.alpha[0]

        if self.normalize:
            prob_volume = F.softmax(cost_volume1, dim=1)
        else:
            prob_volume = cost_volume1

        B, D, H, W = cost_volume1.shape

        if disp_sample is None:
            assert D == self.disp_sample_number, 'The number of disparity samples should be' \
                                                 ' consistent!'
            disp_sample = self.disp_sample.reshape(1,1,1,D).repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            disp_sample = disp_sample.to(cost_volume1.device)
        else:
            assert D == disp_sample.shape[1], 'The number of disparity samples should be' \
                                                 ' consistent!'
        # compute disparity: (BatchSize, 1, Height, Width)
        disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)

        # STAGE 2 
        stage2_disp_sample_pre = disp_map.repeat(1,D//2,1,1).clone()
        stage2_disp_sample_pre += self.disp_sample_pre.reshape(1,1,1,D//2).repeat(B, H, W, 1).permute(0, 3, 1, 2).cuda()
        stage2_disp_sample_behind = disp_map.repeat(1,D//2,1,1).clone()
        stage2_disp_sample_behind += self.disp_sample_behind.reshape(1,1,1,D//2).repeat(B, H, W, 1).permute(0, 3, 1, 2).cuda()
        stage2_disp_sample = torch.stack([stage2_disp_sample_pre, stage2_disp_sample_behind], 1) 
        stage2_disp_sample[d<0] = 0 # mask negative values
        assert stage2_disp_sample.shape[1] == D

        cost_volume2 = self.stage2(cost_volume)
        # scale cost volume with alpha
        cost_volume2 = cost_volume2 * self.alpha[1]

        if self.normalize:
            prob_volume2 = F.softmax(cost_volume2, dim=1)
        else:
            prob_volume2 = cost_volume2
        disp_map2 = torch.sum(prob_volume2 * stage2_disp_sample, dim=1, keepdim=True)

        # STAGE 3
        stage3_disp_sample_pre = disp_map2.repeat(1,D//2,1,1).clone()
        stage3_disp_sample_pre += self.disp_sample_pre.reshape(1,1,1,D//2).repeat(B, H, W, 1).permute(0, 3, 1, 2).cuda()
        stage3_disp_sample_behind = disp_map2.repeat(1,D//2,1,1).clone()
        stage3_disp_sample_behind += self.disp_sample_behind.reshape(1,1,1,D//2).repeat(B, H, W, 1).permute(0, 3, 1, 2).cuda()
        stage3_disp_sample = torch.stack([stage3_disp_sample_pre, stage3_disp_sample_behind], 1) 
        stage3_disp_sample[stage3_disp_sample<0] = 0 # mask negative values
        assert stage3_disp_sample.shape[1] == D

        cost_volume3 = self.stage3(cost_volume)
        # scale cost volume with alpha
        cost_volume3 = cost_volume3 * self.alpha[2]

        if self.normalize:
            prob_volume3 = F.softmax(cost_volume3, dim=1)
        else:
            prob_volume3 = cost_volume3
        disp_map3 = torch.sum(prob_volume3 * stage3_disp_sample, dim=1, keepdim=True)

        return (disp_map, disp_map2, disp_map3)

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Alpha: {}\n'.format(self.alpha)
        repr_str += ' ' * 4 + 'Normalize: {}\n'.format(self.normalize)

        return repr_str

    @property
    def name(self):
        return 'CascadeSoftArgmin'
