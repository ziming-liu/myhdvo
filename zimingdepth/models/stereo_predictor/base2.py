'''
Author: Ziming Liu
Date: 2022-07-08 00:04:48
LastEditors: Ziming
LastEditTime: 2022-09-14 21:17:37
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

@STEREO_PREDICTOR.register_module()
class BaseStereoMatching2(nn.Module):
    """
    Base method. 

    """
    def __init__(self, max_disp, backbone, cost_processor, disp_predictor, num_views=1, batch_norm=True, pretrained=None, **kwargs):
        super(BaseStereoMatching2, self).__init__()
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
        ref_fms, tgt_fms, features = self.backbone(left_img, right_img)
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
                features = features,
            )
        return results