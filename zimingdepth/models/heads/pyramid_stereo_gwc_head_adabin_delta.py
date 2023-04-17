from abc import abstractmethod
import torch.nn as nn
import torch 
import torch.nn.functional as F

#from zimingdepth.models.utils.inverse_warp_3d import inverse_warp_3d
from zimingdepth.models.backbones.psmnet_base  import conv3d_bn, conv3d_bn_relu
from .cost_processors.utils.hourglass import Hourglass,HourglassFPN,HourglassFPN_2plus1D,HourglassFPN_treble1D

from .cost_processors.utils.cat_fms import CAT_FUNCS
from .cost_processors.utils.dif_fms import DIF_FUNCS
#from .cost_processors.utils.correlation1d_cost import COR_FUNCS
from ..builder import build_cost_aggregator,build_loss

from ..registry import HEADS
from .base_stereo_head import BaseStereoHead

from .psmnet_head_48 import *

 

@HEADS.register_module()
class PyramidStereoGWCHead2AdabinDelta(nn.Module):
    def __init__(self, in_channels, max_disp, alpha, normalize, latent_dims=128, 
                 gwc_num_groups=32, cat_channels=64,
    losses=None, learn_resize=None, detach_mono_grad=False,  mask_size=4, if_adabins=True,
      adabins='pixel-wise', **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PyramidStereoGWCHead2AdabinDelta, self).__init__()
        self.in_channels = in_channels
        self.gwc_num_groups = gwc_num_groups
        self.cat_channels = cat_channels

        self.learn_resize = learn_resize
        self.if_adabins = if_adabins
        self.detach_mono_grad = detach_mono_grad
        self.latent_dims = latent_dims
        self.max_disp = max_disp
        self.bins = None
        self.wider_bins = None
        self.disp_loss_func = build_loss(losses)
        self.alpha = alpha
        self.normalize = normalize
        self.batch_norm = True
        self.mask_size = mask_size
        self.adabins = adabins
        self._init_layers(self.batch_norm)

        
        
    def _init_layers(self, batch_norm=True):
        self.out_gwc = nn.Conv2d(self.in_channels[0]//2, self.in_channels[0]//2, kernel_size=1, padding=0, stride=1, bias=False)
        self.out_cat = nn.Conv2d(self.in_channels[0]//2, self.cat_channels//2, kernel_size=1, padding=0, stride=1, bias=False)

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.cat_channels+self.gwc_num_groups, self.latent_dims, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
        )
        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
            conv3d_bn(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False)
        )
        self.dres2 = Hourglass(in_planes=self.latent_dims, batch_norm=batch_norm)
        #self.dres3 = Hourglass(in_planes=self.latent_dims, batch_norm=batch_norm)
        #self.dres4 = Hourglass(in_planes=self.latent_dims, batch_norm=batch_norm)
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
            nn.Conv3d(self.latent_dims, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        #self.classif2 = nn.Sequential(
        #    conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
        #    nn.Conv3d(self.latent_dims, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #)
        #self.classif3 = nn.Sequential(
        #    conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
        #    nn.Conv3d(self.latent_dims, 1, kernel_size=3, stride=1, padding=1, bias=False)
        #)

        self.adabins_conv = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
            nn.Conv3d(self.latent_dims, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def adabins_module(self, adabins, min_disp=None):
        B, D, binH, binW = adabins.shape
        #adabins = (adabins+1)/2
        ##print("prob ")
        #print(adabins[0,:,10,10])
        #adabins = adabins + torch.ones_like(adabins)/D
        #print("dist ")
        #print(adabins[0,:,10,10])
        #ada_dist =  torch.relu(x) + (torch.ones_like(x)/(self.disp_sample_number*4)) # mlp to learn the shifted bin widths
        bin_widths_normed = F.softmax(adabins,1).reshape((B,D,-1)).permute(0,2,1).reshape((-1,D))
        #bin_widths_normed = (adabins / (adabins.sum(-1,True) + 1e-6)).reshape((B,D,-1)).permute(0,2,1).reshape((-1,D))
        #adabins = torch.sigmoid(adabins).reshape((B,D,-1)).permute(0,2,1)
        #bin_widths_normed = adabins / (adabins.sum(-1,True) + 1e-6)
        #print("norm")
        #print(bin_widths_normed[10*10,:])
        bin_widths = D * bin_widths_normed  # BHW D
        #print("D",D)
        #print("norm*D")
        #print(bin_widths[10*10,:])
        if min_disp is None:
            min_disp = -D/2 * torch.ones((B,1,binH,binW), device=adabins.device)
        min_disp = min_disp.reshape((-1,1))
        #print(min_disp.shape)
        #print(bin_widths.shape)
        bin_widths = torch.cat([min_disp, bin_widths], 1) # BHW x (D+1)
        #print(bin_widths)
        #bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=min_disp)
        bin_edges = torch.cumsum(bin_widths, dim=-1) # accumulated sum
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        #print(">> dist >>> ", centers[0][3250])
        #ada_dist = torch.stack([0+(self.disp_channels-0)*(ada_dist[:,:,di]/2+torch.sum(ada_dist[:,:,:di],2)) \
        #                 for di in range(self.disp_channels)], 2)
        #print("ada dist >> ", ada_dist[0][512*10//2])
        centers = centers.reshape((B,-1,D)).permute(0,2,1).reshape((B,D, binH, binW) )
        #print("center")
        #print(centers[0,:,10,10])
        return centers
    def forward(self, feats, bins, y_bins=None):
        if isinstance(feats[0], (list,tuple)):
            B,C,H,W = feats[0][0].shape
        else:
            B,C,H,W = feats[0].shape
        
        
        assert len(feats) == 2, "stereo inputs"
        if isinstance(feats[0], (list,tuple)) and isinstance(feats[1], (list,tuple)) :
            #assert len(stereo_features[0]) == 1
            feats[0],feats[1] = feats[0][0], feats[1][0]
        costs = self.cost_builder(feats,bins, y_bins)
        costs, adabins_feat = self.cost_matcher(costs)
        #min_disps = bins[:,0:1,:,:]
        if not self.if_adabins:
            D = adabins_feat.shape[1]
            bins = torch.linspace(-D//2, D//2-1, D, device=feats[0].device).view(1,D,1,1).repeat(B,1,H,W)
        else:
            bins = self.adabins_module(adabins_feat, ) # update to adaptive bins
        disp = self.disp_predictor(costs, bins)

        #mask = self.mask(feats[0]) # left feature is used to generate weights mask for upsampling
        return disp

    def disp_predictor(self, final_costs, bins):
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
            # compute disparity: (BatchSize, 1, Height, Width)
            disp_map = torch.sum(prob_volume * bins, dim=1, keepdim=True)
            
            pred_disps.append(disp_map)

        return pred_disps
    
    def cost_matcher(self, raw_costs):
        if isinstance(raw_costs, list) or isinstance(raw_costs, tuple):
            assert len(raw_costs) == 1
            raw_costs = raw_costs[0]
        H, W = raw_costs.shape[-2:]
        # raw_cost: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        cost0 = self.dres0(raw_costs)
        cost0 = self.dres1(cost0) # + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        cost1 = self.classif1(out1)
        
        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
        adabins_feat =  self.adabins_conv(out1).squeeze(1)
        
        return cost1, adabins_feat

    def _cost_builder(self, stereo_features, bins, y_bins):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        gwc_cat_cost =  self.fast_gwc_cat_fms(ref_fms,tgt_fms,bins)   #(B, C+G, D, H, W) # 
        return (gwc_cat_cost,)
    
    def cost_builder(self, stereo_features, bins, y_bins):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        gwc_left, gwc_right = self.out_gwc(ref_fms), self.out_gwc(tgt_fms)
        cat_left, cat_right = self.out_cat(ref_fms), self.out_cat(tgt_fms)
 
        gwc_volume = self.build_gwc_volume(gwc_left, gwc_right,  bins, bins.shape[1], self.gwc_num_groups)
        concat_volume = self.build_concat_volume(cat_left, cat_right,  bins, bins.shape[1])
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
    
    def fast_gwc_cat_fms(self, reference_fm, target_fm, bins, y_bins=None):
        B, C, H, W = reference_fm.shape
        D = bins.shape[1]        
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
    
    def loss(self, pred, gt, **kwargs):
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses
