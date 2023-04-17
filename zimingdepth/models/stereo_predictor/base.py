'''
Author: Ziming Liu
Date: 2022-07-08 00:04:48
LastEditors: Ziming Liu
LastEditTime: 2023-03-09 18:22:52
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger

from ..builder import build_backbone, build_cost_processor, build_disp_predictor

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss

@STEREO_PREDICTOR.register_module()
class BaseStereoMatching(nn.Module):
    """
    Base method. 

    """
    def __init__(self, max_disp, backbone, cost_processor, disp_predictor, num_views=1, batch_norm=True, pretrained=None, disp_l1_loss=None, **kwargs):
        """
        disp_l1_loss e.g.
        [dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weight for l1_loss with regard to other loss type
            weight=0.1,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
        )]
        """
        super(BaseStereoMatching, self).__init__()
        self.max_disp = max_disp
        self.num_views = num_views
        self.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        self.cost_processor = build_cost_processor(cost_processor)
        if self.num_views==2:
            self.right_cost_processor = build_cost_processor(cost_processor)

        self.disp_predictor = build_disp_predictor(disp_predictor)
        #if num_views==2:
        #    self.disp_predictor_right = build_disp_predictor(disp_predictor)
        self.disp_l1_loss = disp_l1_loss
        if disp_l1_loss:
            self.disp_l1_loss_func = DispL1Loss(**disp_l1_loss)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
    def forward(self, left_img, right_img):
        # extract image feature
        ref_fms, tgt_fms = self.backbone(left_img, right_img)
        # compute cost volume, return a list
        costs = self.cost_processor(ref_fms, tgt_fms)
        
        # disparity prediction, multi levels
        disps = [self.disp_predictor(cost) for cost in costs]
        #if len(disps)>1: disps = [torch.mean( torch.stack(disps), 0)] # avg 3 levels
        
        disps_right = None
        if self.num_views==2:
             # compute cost volume, return a list
            costs = self.right_cost_processor( ref_fms, tgt_fms,)
            
            # disparity prediction, multi levels
            disps_right = [self.disp_predictor(cost) for cost in costs]
            #if len(disps_right)>1: disps_right = [torch.mean( torch.stack(disps_right), 0)] # avg 3 levels
        results = dict(
                disps=disps,
                right_disps =disps_right,
                costs=costs,
            )
        return results

    def loss(self, pred, gt):
        losses = {}
        if self.disp_l1_loss:
            losses.update(self.disp_l1_loss_func(pred, gt))

        return losses
