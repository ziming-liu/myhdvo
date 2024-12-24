'''
Author: 
Date: 2022-07-07 23:19:self.sparse_channels
LastEditors: Ziming Liu
LastEditTime: 2023-04-04 00:11:03
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
from abc import abstractmethod
import torch.nn as nn
import torch 
import torch.nn.functional as F

from hdvo.models.utils.inverse_warp_3d import inverse_warp_3d
from hdvo.models.backbones.psmnet_base  import conv3d_bn, conv3d_bn_relu
from .cost_processors.utils.hourglass import Hourglass,HourglassFPN,HourglassFPN_2plus1D,HourglassFPN_treble1D

from .cost_processors.utils.cat_fms import CAT_FUNCS
from .cost_processors.utils.dif_fms import DIF_FUNCS
#from .cost_processors.utils.correlation1d_cost import COR_FUNCS
from ..builder import build_cost_aggregator,build_loss

from ..registry import HEADS
from .base_stereo_head import BaseStereoHead


class PSMMatching(nn.Module):
    def __init__(self, in_channels, dense_channels=8, sparse_channels=32, density_ratio=8, fusion_kernel=5, resample_ratio=1, batch_norm=True):
        super().__init__()
        self.resample_ratio = resample_ratio
        self.density_ratio = density_ratio
        self.dense_channels = dense_channels
        self.sparse_channels = sparse_channels
        
        self.dense_dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, in_channels, self.dense_channels, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, self.dense_channels, self.dense_channels, 3, 1, 1, bias=False),
        )
        self.dense_dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.dense_channels, self.dense_channels, 3, 1, 1, bias=False),
            conv3d_bn(batch_norm, self.dense_channels, self.dense_channels, 3, 1, 1, bias=False)
        )
        self.dense_dres2 = Hourglass(in_planes=self.dense_channels, batch_norm=batch_norm)
        self.dense_dres3 = Hourglass(in_planes=self.dense_channels, batch_norm=batch_norm)
        self.dense_dres4 = Hourglass(in_planes=self.dense_channels, batch_norm=batch_norm)
        self.dense_classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.dense_channels, self.dense_channels, 3, 1, 1, bias=False),
            nn.Conv3d(self.dense_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dense_classif2 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.dense_channels, self.dense_channels, 3, 1, 1, bias=False),
            nn.Conv3d(self.dense_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dense_classif3 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.dense_channels, self.dense_channels, 3, 1, 1, bias=False),
            nn.Conv3d(self.dense_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.sparse_dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, in_channels, self.sparse_channels, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, self.sparse_channels, self.sparse_channels, 3, 1, 1, bias=False),
        )
        self.sparse_dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.sparse_channels, self.sparse_channels, 3, 1, 1, bias=False),
            conv3d_bn(batch_norm, self.sparse_channels, self.sparse_channels, 3, 1, 1, bias=False)
        )
        self.sparse_lateral2 = nn.Conv3d(self.sparse_channels+2*self.dense_channels, self.sparse_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sparse_dres2 = Hourglass(in_planes=self.sparse_channels, batch_norm=batch_norm)
        self.sparse_lateral3 = nn.Conv3d(self.sparse_channels+2*self.dense_channels, self.sparse_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sparse_dres3 = Hourglass(in_planes=self.sparse_channels, batch_norm=batch_norm)
        self.sparse_lateral4 = nn.Conv3d(self.sparse_channels+2*self.dense_channels, self.sparse_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sparse_dres4 = Hourglass(in_planes=self.sparse_channels, batch_norm=batch_norm)
        self.sparse_classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.sparse_channels, self.sparse_channels, 3, 1, 1, bias=False),
            nn.Conv3d(self.sparse_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sparse_classif2 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.sparse_channels, self.sparse_channels, 3, 1, 1, bias=False),
            nn.Conv3d(self.sparse_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sparse_classif3 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.sparse_channels, self.sparse_channels, 3, 1, 1, bias=False),
            nn.Conv3d(self.sparse_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )
 
        self.dense_lateral1 = conv3d_bn_relu(batch_norm, self.dense_channels, 2*self.dense_channels, (fusion_kernel,1,1), (self.density_ratio,1,1), ((fusion_kernel-1)//2, 0, 0), bias=False)
        self.dense_lateral2 = conv3d_bn_relu(batch_norm, self.dense_channels, 2*self.dense_channels, (fusion_kernel,1,1), (self.density_ratio,1,1), ((fusion_kernel-1)//2, 0, 0), bias=False)
        self.dense_lateral3 = conv3d_bn_relu(batch_norm, self.dense_channels, 2*self.dense_channels, (fusion_kernel,1,1), (self.density_ratio,1,1), ((fusion_kernel-1)//2, 0, 0), bias=False)

    def forward(self, x):
        dense_x  = F.interpolate(x, scale_factor=(1 / self.resample_ratio,1,1), mode='nearest', )
        #print("dense_x", dense_x.shape)

        sparse_x = F.interpolate(x, scale_factor=(1 / (self.resample_ratio * self.density_ratio),1,1), mode='nearest',)
        #print("sparse_x", sparse_x.shape)
        dense_cost0 = self.dense_dres0(dense_x)
        dense_cost0 = self.dense_dres1(dense_cost0) + dense_cost0
        #print("dense_cost0", dense_cost0.shape)
        sparse_cost0 = self.sparse_dres0(sparse_x)
        sparse_cost0 = self.sparse_dres1(sparse_cost0) + sparse_cost0
        #print("sparse_cost0", sparse_cost0.shape)
        dense_cost0_lateral = self.dense_lateral1(dense_cost0)
        sparse_cost0 = torch.cat([sparse_cost0, dense_cost0_lateral], dim=1) # self.sparse_channels + self.sparse_channels C 
        sparse_cost0 = self.sparse_lateral2(sparse_cost0)
        sparse_out1, sparse_pre1, sparse_post1 = self.sparse_dres2(sparse_cost0, None, None)
        sparse_out1 = sparse_out1 + sparse_cost0

        dense_out1, dense_pre1, dense_post1 = self.dense_dres2(dense_cost0, None, None)
        dense_out1 = dense_out1 + dense_cost0
        #print("dense_out1", dense_out1.shape)
        dense_out1_lateral = self.dense_lateral2(dense_out1)
        #print("dense_out1_lateral", dense_out1_lateral.shape)
        sparse_out1_cat = torch.cat([sparse_out1, dense_out1_lateral], dim=1) # self.sparse_channels + self.sparse_channels + self.sparse_channels C
        sparse_out1_cat = self.sparse_lateral3(sparse_out1_cat)
        sparse_out2, sparse_pre2, sparse_post2 = self.sparse_dres3(sparse_out1_cat, None, None)
        sparse_out2 = sparse_out2 + sparse_out1_cat

        dense_out2, dense_pre2, dense_post2 = self.dense_dres3(dense_out1,  None, None)
        dense_out2 = dense_out2 + dense_out1

        dense_out2_lateral = self.dense_lateral3(dense_out2)
        sparse_out2_cat = torch.cat([sparse_out2, dense_out2_lateral], dim=1) # self.sparse_channels + self.sparse_channels + self.sparse_channels + self.sparse_channels C
        sparse_out2_cat = self.sparse_lateral4(sparse_out2_cat)
        sparse_out3, sparse_pre3, sparse_post3 = self.sparse_dres4(sparse_out2_cat, None, None)
        sparse_out3 = sparse_out3 + sparse_out2_cat

        dense_out3, dense_pre3, dense_post3 = self.dense_dres4(dense_out2,  None, None)
        dense_out3 = dense_out3 + dense_out2

        sparse_cost1 = self.sparse_classif1(sparse_out1)
        sparse_cost2 = self.sparse_classif2(sparse_out2) + sparse_cost1
        sparse_cost3 = self.sparse_classif3(sparse_out3) + sparse_cost2

        dense_cost1 = self.dense_classif1(dense_out1)
        dense_cost2 = self.dense_classif2(dense_out2) + dense_cost1
        dense_cost3 = self.dense_classif3(dense_out3) + dense_cost2

         
        return sparse_cost1, sparse_cost2, sparse_cost3, dense_cost1, dense_cost2, dense_cost3


# Concatenate left and right feature to form cost volume
@HEADS.register_module()
class DenseSparseHead(nn.Module):
    def __init__(self, in_channels, dense_disp_range, sparse_disp_range, pred_disp_range, 
                  alpha, normalize, losses=None, dense_channels=8, sparse_channels=32,
                  local_predictor=False, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(DenseSparseHead, self).__init__()
        self.in_channels = in_channels
        self.dense_disp_range = dense_disp_range
        self.sparse_disp_range = sparse_disp_range
        self.pred_disp_range = pred_disp_range

        self.dense_start_idx = dense_disp_range[0]
        self.dense_end_idx = dense_disp_range[1] - dense_disp_range[2]
        self.dense_max_disp = dense_disp_range[1]
        self.dense_interval = dense_disp_range[2]
        self.dense_num_sampler = self.dense_max_disp // self.dense_interval
        self.dense_sampler = torch.linspace(self.dense_start_idx, self.dense_end_idx,\
                                             self.dense_num_sampler, device='cuda')

        self.sparse_start_idx = sparse_disp_range[0]
        self.sparse_end_idx = sparse_disp_range[1] - sparse_disp_range[2]
        self.sparse_max_disp = sparse_disp_range[1]
        self.sparse_interval = sparse_disp_range[2]
        self.sparse_num_sampler = self.sparse_max_disp // self.sparse_interval
        self.sparse_sampler = torch.linspace(self.sparse_start_idx, self.sparse_end_idx,\
                                             self.sparse_num_sampler, device='cuda')

        self.pred_start_idx = pred_disp_range[0]
        self.pred_end_idx = pred_disp_range[1] - pred_disp_range[2]
        self.pred_max_disp = pred_disp_range[1]
        self.pred_interval = pred_disp_range[2]
        self.pred_num_sampler = self.pred_max_disp // self.pred_interval
        self.pred_sampler = torch.linspace(self.pred_start_idx, self.pred_end_idx,\
                                           self.pred_num_sampler, device='cuda')
        


        self.alpha = alpha
        self.normalize = normalize

        if losses is not None:
            self.disp_loss_func = build_loss(losses)
            
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
        
        self.matching_net = PSMMatching(in_channels[0], density_ratio=self.dense_num_sampler//self.sparse_num_sampler, \
                                        dense_channels=dense_channels, sparse_channels=sparse_channels)
        
        self.conv_fuse1 = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fuse2 = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_fuse3 = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)

    

    def forward(self, stereo_features):
        raw_costs = self.cost_builder(stereo_features)
        costs_volume = self.cost_matcher(raw_costs)
        if self.local_predictor:
            pred_disps = self.local_disp_predictor(costs_volume)
        else:
            pred_disps = self.disp_predictor(costs_volume)
        #for i in range(len(pred_disps)):
        #    print(max(pred_disps[i].view(-1)))
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

            assert D == self.pred_num_sampler, 'The number of disparity samples should be  consistent!'
            pred_sampler = self.pred_sampler.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

        
            # compute disparity: (BatchSize, 1, Height, Width)
            disp_map = torch.sum(prob_volume * pred_sampler, dim=1, keepdim=True)
            
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
            assert D == self.pred_num_sampler, 'Number of disparity sample should be same' \
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
        # raw_cost: (BatchSize, Channels*2, denseD/4, Height/4, Width/4)
        sparse_cost1, sparse_cost2, sparse_cost3, dense_cost1, dense_cost2, dense_cost3 = \
            self.matching_net(raw_costs)

        full_h, full_w = H * 4, W * 4
        full_D = self.pred_num_sampler 
        sparse_cost1 = F.interpolate(sparse_cost1, [full_D, full_h, full_w],mode='trilinear', align_corners=False)
        sparse_cost2 = F.interpolate( sparse_cost2, [full_D, full_h, full_w],mode='trilinear', align_corners=False)
        sparse_cost3 = F.interpolate( sparse_cost3, [full_D, full_h, full_w], mode='trilinear', align_corners=False)

        dense_cost1 = F.interpolate(dense_cost1, [full_D, full_h, full_w],mode='trilinear', align_corners=False)
        dense_cost2 = F.interpolate( dense_cost2, [full_D, full_h, full_w],mode='trilinear', align_corners=False)
        dense_cost3 = F.interpolate( dense_cost3, [full_D, full_h, full_w], mode='trilinear', align_corners=False)

        cost3 = self.conv_fuse3(torch.cat([sparse_cost3, dense_cost3], dim=1)).squeeze(1)
        cost2 = self.conv_fuse2(torch.cat([sparse_cost2, dense_cost2], dim=1)).squeeze(1)
        cost1 = self.conv_fuse1(torch.cat([sparse_cost1, dense_cost1], dim=1)).squeeze(1)
        #print("cost3.shape", cost3.shape)
        return (cost3, cost2, cost1)
 

    def cost_builder(self, stereo_features):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        cat_cost = self.cat_fms(ref_fms, tgt_fms,)
        return (cat_cost,)

    def cat_fms(self, reference_fm, target_fm):
        """
        Concat left and right in Channel dimension to form the raw cost volume.
        Inputs:
            reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
            target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

        Output:
            concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, dense_num_sampler, Height, Width] layout

        """
        device = reference_fm.device
        N, C, H, W = reference_fm.shape
        concat_fm = torch.zeros(N, C * 2, self.dense_num_sampler, H, W).cuda() # fix the type bug when using half-float. ziming 21-7-self.dense_channels
        idx = 0
        for i in self.dense_sampler:
            i = i.long() # convert torch.Tensor to int, so that it can be index
            if i > 0:
                concat_fm[:, :C, idx, :, i:] = reference_fm[:, :, :, i:]
                concat_fm[:, C:, idx, :, i:] = target_fm[:, :, :, :-i]
            elif i == 0:
                concat_fm[:, :C, idx, :, :] = reference_fm
                concat_fm[:, C:, idx, :, :] = target_fm
            else:
                concat_fm[:, :C, idx, :, :i] = reference_fm[:, :, :, :i]
                concat_fm[:, C:, idx, :, :i] = target_fm[:, :, :, abs(i):]
            idx = idx + 1

        concat_fm = concat_fm.contiguous()
        return concat_fm


    def fast_cat_fms(self, reference_fm, target_fm):
        device = reference_fm.device
        B, C, H, W = reference_fm.shape
        D = self.dense_num_sampler
        dense_sampler = self.dense_sampler.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -dense_sampler.float(), padding_mode='zeros')

        # mask out features in reference
        #concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-self.dense_channels

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm

    def loss(self, pred, gt, **kwargs):
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses

