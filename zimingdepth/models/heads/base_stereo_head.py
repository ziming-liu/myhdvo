'''
Author: 
Date: 2022-07-07 23:19:32
LastEditors: Ziming Liu
LastEditTime: 2023-03-16 11:23:07
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
from abc import abstractmethod
import torch.nn as nn
import torch 
import torch.nn.functional as F

from zimingdepth.models.utils.inverse_warp_3d import inverse_warp_3d

from .cost_processors.utils.cat_fms import CAT_FUNCS
from .cost_processors.utils.dif_fms import DIF_FUNCS
#from .cost_processors.utils.correlation1d_cost import COR_FUNCS
from ..builder import  build_loss

from ..registry import HEADS

# Concatenate left and right feature to form cost volume
@HEADS.register_module()
class BaseStereoHead(nn.Module):

    def __init__(self, in_channels, disp_range,  alpha, normalize, losses=None, stereo_feat_cons_losses=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(BaseStereoHead, self).__init__()
        self.in_channels = in_channels
        self.disp_range = disp_range 
        self.start_disp = disp_range[0]
        self.max_disp = disp_range[1]
        self.end_disp = disp_range[1]-1
        self.dilation = disp_range[2]
        self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation
        # generate disparity sample, in [disp_sample_number,] layout
        self.disp_sample = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        )

        self.alpha = alpha
        self.normalize = normalize

        if losses is not None:
            self.disp_loss_func = build_loss(losses)
        self.stereo_feat_cons_losses = stereo_feat_cons_losses
        if stereo_feat_cons_losses is not None:
            self.stereo_feat_cons_losses_func = build_loss(stereo_feat_cons_losses)

    @abstractmethod
    def disp_predictor(self, final_costs):
        pass
    
    @abstractmethod
    def cost_matcher(self, costs):
        pass

    @abstractmethod
    def cost_builder(self, stereo_features):
        pass

    def forward(self, stereo_features):
        raw_costs = self.cost_builder(stereo_features)
        decoded_features = self.cost_matcher(raw_costs)
        pred_disps = self.disp_predictor(decoded_features)

        return pred_disps


    def loss(self, pred, gt, **kwargs):
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses

    def loss_rectify_calib(self, left_feat, right_feat, left_gtdisp=None, right_gtdisp=None, **kwargs):
        losses = {}
        losses.update(self.stereo_feat_cons_losses_func(left_feat, right_feat, left_gtdisp, right_gtdisp, **kwargs))
        return losses
