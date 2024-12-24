'''
Author: 
Date: 2022-07-07 23:19:32
LastEditors: Ziming Liu
LastEditTime: 2023-03-16 11:23:17
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
from .base import CostProcessor
import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from hdvo.models.utils.inverse_warp_3d import inverse_warp_3d
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.cnn import build_norm_layer

from .utils.cat_fms import CAT_FUNCS
from .utils.dif_fms import DIF_FUNCS
#from .utils.correlation1d_cost import COR_FUNCS
from ...builder import build_cost_aggregator

from ...registry import COST_PROCESSORS


class CrossAttentionLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super(CrossAttentionLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x1, x2):

        def _inner_forward(x1, x2):
            x1 = self.attn(self.norm1(x1), self.norm1(x2), self.norm1(x1), identity=x1)
            x1 = self.ffn(self.norm2(x1), identity=x1)
            return x1

        if self.with_cp and x1.requires_grad and x2.requires_grad:
            x1 = cp.checkpoint(_inner_forward, x1, x2)
        else:
            x1 = _inner_forward(x1,x2)
        return x1


# Concatenate left and right feature to form cost volume
@COST_PROCESSORS.register_module()
class CrossAttenCostProcessor(CostProcessor):

    def __init__(self, cost_computation=None, cost_aggregator=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(CrossAttenCostProcessor, self).__init__()
        self.attention =  CrossAttentionLayer(**kwargs)

        #self.aggregator = build_cost_aggregator(cost_aggregator)


    def cross_atten_fms(self, reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
        """
        Concat left and right in Channel dimension to form the raw cost volume.
        Args:
            max_disp, (int): under the scale of feature used,
                often equals to (end disp - start disp + 1), the maximum searching range of disparity
            start_disp (int): the start searching disparity index, usually be 0
                dilation (int): the step between near disparity index
            dilation (int): the step between near disparity index

        Inputs:
            reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
            target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

        Output:
            concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, disp_sample_number, Height, Width] layout

        """
        device = reference_fm.device
        N, C, H, W = reference_fm.shape
        print(reference_fm.shape)
        reference_fm = reference_fm.view(N,C,H*W).permute(2,0,1)
        target_fm    =  target_fm.view(N,C,H*W).permute(2,0,1)
        print("reference_fm>>",reference_fm.shape)
        atten_cost = self.attention(reference_fm , target_fm)
        print(atten_cost.shape)
        atten_cost = atten_cost.contiguous()
        return atten_cost

 
        
    def forward(self, ref_fms, tgt_fms, disp_sample=None):
        costs = self.cross_atten_fms(ref_fms, tgt_fms,)
         
        # agregator has been FFN layer in attention module
        # 2. aggregate cost by 3D-hourglass
        #costs = self.aggregator(cat_cost)

        return [costs]

