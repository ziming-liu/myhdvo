"""
GWC-Net Stereo Matching Head.

This module implements the Group-wise Correlation (GWC) stereo matching head
with hourglass cost aggregation.

Reference: https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark

Author: Ziming Liu
Date: 2022-07-07
Last Modified: 2023-03-29
"""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdvo.models.backbones.psmnet_base import conv3d_bn, conv3d_bn_relu

from ..builder import build_cost_aggregator, build_loss
from ..registry import HEADS
from .base_stereo_head import BaseStereoHead
from .cost_processors.utils.hourglass import Hourglass
from ..stereo_predictor.cascade_stereo_gwcnet import *  # noqa: F401, F403



# Concatenate left and right feature to form cost volume
@HEADS.register_module()
class GWCNetHead(BaseStereoHead):
    """Group-wise Correlation Network head for stereo matching.
    
    This head implements group-wise correlation stereo matching with
    hourglass-based cost aggregation and optional local disparity prediction.
    
    Args:
        in_channels (int): Number of input feature channels.
        disp_range (tuple): Disparity range (start, max, dilation).
        alpha (float): Scaling factor for cost volume.
        normalize (bool): Whether to apply softmax normalization.
        gwc_num_groups (int): Number of groups for group-wise correlation.
            Defaults to 32.
        cat_channels (int): Number of channels for concatenation features.
            Defaults to 64.
        losses (dict, optional): Loss function configuration.
        local_predictor (bool): Whether to use local disparity predictor.
            Defaults to False.
        **kwargs: Additional arguments passed to BaseStereoHead.
    """
    
    def __init__(self, in_channels, disp_range, alpha, normalize,
                   gwc_num_groups=32, cat_channels=64, losses=None, local_predictor=False, **kwargs):
        super(GWCNetHead, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
        self.gwc_num_groups = gwc_num_groups
        self.cat_channels = cat_channels

        self.disp_range = disp_range 
        self.start_disp = disp_range[0]
        self.max_disp = disp_range[1]
        self.end_disp = disp_range[1]-1
        self.dilation = disp_range[2]
        self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation
        # generate disparity sample, in [disp_sample_number,] layout
        self.disp_sample = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        ).cuda()
        self.disp_sample_pred_layer = torch.linspace(
            self.start_disp, disp_range[1]*4-1, (self.max_disp*4 + self.dilation - 1) // self.dilation
        ).cuda()

        # the radius of window when local sampling
        self.radius=3
        # the start disparity of disparity search range
        #start_disp=0,
        # the step between near disparity sample
        #dilation=1,
        # the step between near disparity index when local sampling
        self.radius_dilation=1
        self.local_predictor = local_predictor

        self.alpha = alpha
        self.normalize = normalize
        self.batch_norm = True
        self._init_layers(self.batch_norm)
        
    def _init_layers(self, batch_norm=True):
        self.out_gwc = nn.Conv2d(self.in_channels[0]//2, self.in_channels[0]//2, kernel_size=1, padding=0, stride=1, bias=False)
        self.out_cat = nn.Conv2d(self.in_channels[0]//2, self.cat_channels//2, kernel_size=1, padding=0, stride=1, bias=False)

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.cat_channels+self.gwc_num_groups, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
        )
        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            conv3d_bn(batch_norm, 32, 32, 3, 1, 1, bias=False)
        )
        self.dres2 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres3 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres4 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, stereo_features):
        raw_costs = self.cost_builder(stereo_features)
        
        decoded_features = self.cost_matcher(raw_costs)
        if self.local_predictor:
            pred_disps = self.local_disp_predictor(decoded_features)
        else:
            pred_disps = self.disp_predictor(decoded_features)

        return pred_disps


    def disp_predictor(self, final_costs):
        if not isinstance(final_costs, list) and not isinstance(final_costs, tuple):
            final_costs= [final_costs]
        pred_disps = []
        for i in range(len(final_costs)):
            cost_volume = final_costs[i]
            if cost_volume.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)'
                                .format(cost_volume.dim()))

            # scale cost volume with alpha
            cost_volume = cost_volume * self.alpha

            if self.normalize:
                prob_volume = F.softmax(cost_volume, dim=1)
            else:
                prob_volume = cost_volume

            B, D, H, W = cost_volume.shape

            assert D == self.disp_sample_number*4, 'The number of disparity samples should be' \
                                                ' consistent!'
            disp_sample = self.disp_sample_pred_layer.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            disp_sample = disp_sample.to(cost_volume.device)

        
            # compute disparity: (BatchSize, 1, Height, Width)
            disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)
            
            pred_disps.append(disp_map)

        return pred_disps


    def local_disp_predictor(self, final_costs, disp_sample=None):
        if not isinstance(final_costs, list) and not isinstance(final_costs, tuple):
            final_costs= [final_costs]
        pred_disps = []
        for i in range(len(final_costs)):
            cost_volume = final_costs[i]
            # note, cost volume direct represent similarity
            # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

            # grab index with max similarity

            B = cost_volume.size()[0]

            D = cost_volume.size()[1]
            assert D == self.disp_sample_number*4, 'Number of disparity sample should be same' \
                                                'with predicted disparity number in cost volume!'

            H = cost_volume.size()[2]
            W = cost_volume.size()[3]

            # d':|d'-d|<=sigma, d' = argmax( C(d) for d in dim[1] ), (BatchSize, 1, Height, Width)
            # it's only the index for array, not real disparity index
            max_index = torch.argmax(cost_volume, dim=1, keepdim=True)

            # sample near the index of max similarity, get [2 * radius + 1]
            # for example, if dilation=2, disp_sample_radius =2, we will get (-4, -2, 0, 2, 4)
            interval = torch.linspace(-self.radius * self.radius_dilation,
                                    self.radius * self.radius_dilation,
                                    2 * self.radius + 1).long().to(cost_volume.device)
            # (BatchSize, 2 * radius + 1, Height, Width)
            interval = interval.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

            # (BatchSize, 2*radius+1, Height, Width)
            index_group = (max_index + interval)


            # get mask in [0, D-1],
            # (BatchSize, 2*radius+1, Height, Width)
            mask = ((index_group >= 0) & (index_group <= D-1)).detach().type_as(cost_volume)
            index_group = index_group.clamp(0, D-1)

            # gather values in cost_volume which index = index_group,
            # (BatchSize, 2*radius+1, Height, Width)
            gathered_cost_volume = torch.gather(cost_volume, dim=1, index=index_group)

            # convert index_group from torch.LongTensor to torch.FloatTensor
            index_group = index_group.type_as(cost_volume)

            # convert to real disparity sample index
            disp_sample = self.start_disp + index_group * self.dilation

            # d * P(d), and mask out index out of (start_disp, end_disp), (BatchSize, 1, Height, Width)
            # if index in (start_disp, end_disp), keep the original disparity value, otherwise -10000.0, as e(-10000.0) approximate 0.0
            # scale cost volume with alpha
            gathered_cost_volume = gathered_cost_volume * self.alpha

            # (BatchSize, 2 * radius + 1, Height, Width)
            gathered_prob_volume = F.softmax((gathered_cost_volume * mask + (1 - mask) * (-10000.0 * self.alpha)), dim=1)

            # (BatchSize, 1, Height, Width)
            disp_map = (gathered_prob_volume * disp_sample).sum(dim=1, keepdim=True)

            pred_disps.append(disp_map)

        return pred_disps

    def cost_matcher(self, raw_costs):
        if isinstance(raw_costs, list) or isinstance(raw_costs, tuple):
            assert len(raw_costs) == 1
            raw_costs = raw_costs[0]
        H, W = raw_costs.shape[-2:]
        # raw_cost: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        cost0 = self.dres0(raw_costs)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre2, post2)
        out3 = out3 + cost0
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        # (BatchSize, 1, max_disp, Height, Width)
        full_h, full_w = H * 4, W * 4
        align_corners = True
        cost1 = F.interpolate(
            cost1, [self.max_disp*4, full_h, full_w],
            mode='trilinear', align_corners=align_corners
        )
        cost2 = F.interpolate(
            cost2, [self.max_disp*4, full_h, full_w],
            mode='trilinear', align_corners=align_corners
        )
        cost3 = F.interpolate(
            cost3, [self.max_disp*4, full_h, full_w],
            mode='trilinear', align_corners=align_corners
        )

        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
        cost2 = torch.squeeze(cost2, 1)
        cost3 = torch.squeeze(cost3, 1)

        return (cost3, cost2, cost1)

    def _cost_builder(self, stereo_features,  ):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        gwc_cat_cost =  self.fast_gwc_cat_fms(ref_fms,tgt_fms)   #(B, C+G, D, H, W) # 
        return (gwc_cat_cost,)
    
    def cost_builder(self, stereo_features,  ):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        gwc_left, gwc_right = self.out_gwc(ref_fms), self.out_gwc(tgt_fms)
        cat_left, cat_right = self.out_cat(ref_fms), self.out_cat(tgt_fms)
 
        gwc_volume = self.build_gwc_volume(gwc_left, gwc_right,  self.disp_sample, self.max_disp, self.gwc_num_groups)
        concat_volume = self.build_concat_volume(cat_left, cat_right,  self.disp_sample, self.max_disp)
        gwc_cat_cost = torch.cat((gwc_volume, concat_volume), 1)   #(B, C+G, D, H, W) # 
        return (gwc_cat_cost,)
    

    def get_warped_feats(self, x, y, disp_range_samples, ndisp):
        if len(disp_range_samples.shape)!=4:
            disp_range_samples = disp_range_samples.reshape(1,ndisp,1,1)
        bs, channels, height, width = y.size()

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = mw - disp_range_samples

        # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
        # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

        y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros').view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)


        # a littel difference, no zeros filling
        x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
        x_warped = x_warped.transpose(0, 1) #(C, B, D, H, W)
        #x1 = x2 + d >= d
        x_warped[:, mw < disp_range_samples] = 0
        x_warped = x_warped.transpose(0, 1) #(B, C, D, H, W)

        return x_warped, y_warped

    def build_concat_volume(self, x, y, disp_range_samples, ndisp):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        concat_cost = x.new().resize_(bs, channels * 2, ndisp, height, width).zero_()  # (B, 2C, D, H, W)

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp)
        concat_cost[:, x.size()[1]:, :, :, :] = y_warped
        concat_cost[:, :x.size()[1], :, :, :] = x_warped

        return concat_cost

    def build_gwc_volume(self, x, y, disp_range_samples, ndisp, gwc_num_groups):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp) #(B, C, D, H, W)

        assert channels % gwc_num_groups == 0
        channels_per_group = channels // gwc_num_groups
        gwc_cost = (x_warped * y_warped).view([bs, gwc_num_groups, channels_per_group, ndisp, height, width]).mean(dim=2)  #(B, G, D, H, W)

        return gwc_cost
    

    def fast_gwc_cat_fms(self, reference_fm, target_fm, y_bins=None ):
        B, C, H, W = reference_fm.shape
        D = self.max_disp
        bins = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).type_as(reference_fm)
        
        reference_fm_gwc, target_fm_gwc = reference_fm, target_fm # self.out_gwc(reference_fm), self.out_gwc(target_fm)
        reference_fm_cat, target_fm_cat = self.out_cat(reference_fm_gwc), self.out_cat(target_fm_gwc)
        # expand D dimension
        concat_reference_fm = reference_fm_cat.unsqueeze(2).expand(B, reference_fm_cat.shape[1], D, H, W)
        concat_target_fm = target_fm_cat.unsqueeze(2).expand(B, target_fm_cat.shape[1], D, H, W)
        gwc_reference_fm =  reference_fm_gwc.unsqueeze(2).expand(B, reference_fm_gwc.shape[1], D, H, W)
        gwc_target_fm =  target_fm_gwc.unsqueeze(2).expand(B, target_fm_gwc.shape[1], D, H, W)

        # GWC
        gwc_target_fm = inverse_warp_3d(gwc_target_fm.float(), -bins.float(), padding_mode='zeros', disp_Y=y_bins)
        gwc_reference_fm = gwc_reference_fm * (gwc_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8
        assert C % self.gwc_num_groups == 0
        channels_per_group = C // self.gwc_num_groups
        gwc_cost = (gwc_reference_fm * gwc_target_fm).view([B, self.gwc_num_groups, channels_per_group, D, H, W]).mean(dim=2)  #(B, G, D, H, W)

        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -bins.float(), padding_mode='zeros', disp_Y=y_bins)
        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8
        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        
        gwc_cat_cost = torch.cat((gwc_cost, concat_fm), 1)   #(B, C+G, D, H, W) # 
        return gwc_cat_cost