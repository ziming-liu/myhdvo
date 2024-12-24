'''
Author: 
Date: 2022-07-07 23:19:32
LastEditors: Ziming Liu
LastEditTime: 2023-03-16 11:23:15
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
from .base import CostProcessor
import torch.nn as nn
import torch 
import torch.nn.functional as F

from hdvo.models.utils.inverse_warp_3d import inverse_warp_3d

from .utils.cat_fms import CAT_FUNCS
from .utils.dif_fms import DIF_FUNCS
#from .utils.correlation1d_cost import COR_FUNCS
from ...builder import build_cost_aggregator
from mmcv.cnn import ConvModule, constant_init, kaiming_init

from ...registry import COST_PROCESSORS


# Concatenate left and right feature to form cost volume
@COST_PROCESSORS.register_module()
class CatSimple2DCostProcessor(CostProcessor):

    def __init__(self,  cost_aggregator, cost_computation=None, num_level=3, max_disp=192,conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True), align_corners=False, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(CatSimple2DCostProcessor, self).__init__()
        self.cat_func = cost_computation.get('type', 'default')
        self.default_args = cost_computation.copy()
        self.default_args.pop('type')
        self.align_corners = align_corners
        self.aggregator = build_cost_aggregator(cost_aggregator)

        
 
    def forward(self, ref_fms, tgt_fms, disp_sample=None):
        # 1. build raw cost by concat
 
        cat_cost = torch.stack([ref_fms, tgt_fms], 2) # stack, b, c, 2, h, w
        #print(cat_cost.shape)
        # 2. aggregate cost by 3D-hourglass
        costs = self.aggregator(cat_cost)
        
        return costs

