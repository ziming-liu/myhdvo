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
class PyramidStereoHead2AdabinDelta(nn.Module):
    def __init__(self, in_channels, max_disp, alpha, normalize, latent_dims=128, 
    losses=None, learn_resize=None, detach_mono_grad=False,  mask_size=4, if_adabins=True,
      adabins='pixel-wise', **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PyramidStereoHead2AdabinDelta, self).__init__()
        self.in_channels = in_channels
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
        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.in_channels[0], self.latent_dims, 3, 1, 1, bias=False),
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

    def cost_builder(self, stereo_features,bins, y_bins):
        
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        #print(ref_fms.shape)
        cat_cost = self.fast_cat_fms(ref_fms, tgt_fms,bins, y_bins)
        return (cat_cost,)

    def cat_fms(self, reference_fm, target_fm):
        """
        Concat left and right in Channel dimension to form the raw cost volume.
        Inputs:
            reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
            target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

        Output:
            concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, disp_sample_number, Height, Width] layout

        """
        device = reference_fm.device
        N, C, H, W = reference_fm.shape
        concat_fm = torch.zeros(N, C * 2, self.disp_sample_number, H, W).cuda() # fix the type bug when using half-float. ziming 21-7-8
        idx = 0
        for i in self.disp_sample:
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
        
    def fast_cat_fms(self, reference_fm, target_fm, bins, y_bins=None):
        B, C, H, W = reference_fm.shape
        D = bins.shape[1]
        #disp_sample = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -bins.float(), padding_mode='zeros', disp_Y=y_bins)

        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm

    def loss(self, pred, gt, **kwargs):
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses




def inverse_warp_3d(img, disp, padding_mode='zeros', disp_Y=None):
    """
    Args:
        img (Tensor): the source image (where to sample pixels) -- [B, C, H, W] or [B, C, D, H, W]
        disp (Tensor): disparity map of the target image -- [B, D, H, W]
        padding_mode (str): padding mode, default is zero padding
        disp_Y (Tensor): disparity map of the target image along Y-axis -- [B, D, H, W]
    Returns:
        projected_img (Tensor): source image warped to the target image -- [B, C, D, H, W]
    """

    device = disp.device
    B, D, H, W = disp.shape
    C = img.shape[1]

    if disp_Y is not None:
        assert disp.shape == disp_Y.shape, 'disparity map along x and y axis should have same shape!'
    if img.dim() == 4:
        img = img.unsqueeze(2).expand(B, C, D, H, W)
    elif img.dim() == 5:
        assert D == img.shape[2], 'The disparity number should be same between image and disparity map!'
    else:
        raise ValueError('image is only allowed with 4 or 5 dimensions, '
                         'but got {} dimensions!'.format(img.dim()))

    # get mesh grid for each dimension
    grid_d = torch.linspace(0, D - 1, D).view(1, D, 1, 1).expand(B, D, H, W).to(device)
    grid_h = torch.linspace(0, H - 1, H).view(1, 1, H, 1).expand(B, D, H, W).to(device)
    grid_w = torch.linspace(0, W - 1, W).view(1, 1, 1, W).expand(B, D, H, W).to(device)

    # shift the index of W dimension with disparity
    grid_w = grid_w + disp
    if disp_Y is not None:
        grid_h = grid_h + disp_Y

    # normalize the grid value into [-1, 1]; (0, D-1), (0, H-1), (0, W-1)
    grid_d = (grid_d / (D - 1) * 2) - 1
    grid_h = (grid_h / (H - 1) * 2) - 1
    grid_w = (grid_w / (W - 1) * 2) - 1

    # concatenate the grid_* to [B, D, H, W, 3]
    grid_d = grid_d.unsqueeze(4)
    grid_h = grid_h.unsqueeze(4)
    grid_w = grid_w.unsqueeze(4)
    grid = torch.cat((grid_w, grid_h, grid_d), 4)

    projected_img = F.grid_sample(img, grid, padding_mode=padding_mode, align_corners=False)

    return projected_img

@HEADS.register_module()
class PyramidStereoHead2AdabinDeltaLargeRange(PyramidStereoHead2AdabinDelta):
    def __init__(self, in_channels, max_disp, alpha, normalize, latent_dims=128, losses=None, learn_resize=None, detach_mono_grad=False, mask_size=4, adabins='pixel-wise', **kwargs):
        super().__init__(in_channels, max_disp, alpha, normalize, latent_dims, losses, learn_resize, detach_mono_grad, mask_size, adabins, **kwargs)
    
        
    def _init_layers(self, batch_norm=True):
        self.conv0 = nn.Conv3d(self.in_channels[0], self.latent_dims, 1, 1, 0, bias=False)
        self.down0 = nn.Conv3d(self.latent_dims, self.latent_dims, (2, 1, 1), (2, 1, 1), 0, bias=False)
        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
        )
        self.down1 = nn.Conv3d(self.latent_dims, self.latent_dims, (2, 1, 1), (2, 1, 1), 0, bias=False)
        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
            conv3d_bn(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False)
        )
        self.dres2 = Hourglass(in_planes=self.latent_dims, batch_norm=batch_norm)
        #self.dres3 = Hourglass(in_planes=self.latent_dims, batch_norm=batch_norm)
        #self.dres4 = Hourglass(in_planes=self.latent_dims, batch_norm=batch_norm)
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.latent_dims, self.latent_dims, 3, 1, 1, bias=False),
            nn.Conv3d(self.latent_dims, 1, kernel_size=1, stride=1, padding=0, bias=False),
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
            nn.Conv3d(self.latent_dims, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def cost_matcher(self, raw_costs):
        if isinstance(raw_costs, list) or isinstance(raw_costs, tuple):
            assert len(raw_costs) == 1
            raw_costs = raw_costs[0]
        H, W = raw_costs.shape[-2:]
        # raw_cost: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        raw_costs = self.conv0(raw_costs)
        cost0 = self.down0(raw_costs)
        cost0 = self.dres0(cost0)
        cost1 = self.down1(cost0)
        cost1 = self.dres1(cost1) # + cost0
        out1, pre1, post1 = self.dres2(cost1, None, None)
        out1 = F.interpolate(out1, scale_factor=(2,1,1), mode="trilinear", align_corners=False) + cost0
        out2 = F.interpolate(out1, scale_factor=(2,1,1), mode="trilinear", align_corners=False) + raw_costs


        costs = self.classif1(out2).squeeze(1)
        adabins_feat =  self.adabins_conv(out2).squeeze(1)
        
        return costs, adabins_feat
