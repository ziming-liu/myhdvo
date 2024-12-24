'''
Developer: ACENTAURI team, INRIA institute
Author: 
Date: 2023-06-24 22:53:29
LastEditors: Ziming Liu
LastEditTime: 2023-10-04 22:46:47
Description: refer to https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py
'''

import torch
import torch.nn as nn
from collections import OrderedDict

from ..registry import HEADS

@HEADS.register_module()
class PoseNetHead(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseNetHead, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        #last_features = [f[-1] for f in input_features]
         
        #cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        #cat_features = torch.cat(cat_features, 1)
        out = self.relu(self.convs["squeeze"](input_features[-1]))
        #out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        #out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        out = out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

    def loss(self, axisangle, translation, axisangle_gt, translation_gt, weights=None):
       '''
       Description: Supervised pose loss
       Args:: 
       Returns:: 
       '''        
       if weights is None:
           weights = [100, 100, 100, 1, 1, 1]
       assert len(weights) == 6
       weights = torch.tensor(weights).to(axisangle.device).float()
       loss = torch.mean(weights * torch.abs(axisangle - axisangle_gt) ** 2) + torch.mean(weights * torch.abs(translation - translation_gt) ** 2)
       return loss