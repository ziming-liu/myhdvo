'''
Author: 
Date: 2022-07-07 23:19:32
LastEditors: Ziming Liu
LastEditTime: 2023-03-16 11:23:09
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
from abc import abstractmethod
import torch.nn as nn
import torch 
import torch.nn.functional as F

from zimingdepth.models.utils.inverse_warp_3d import inverse_warp_3d
from zimingdepth.models.backbones.psmnet_base  import conv3d_bn, conv3d_bn_relu
from .cost_processors.utils.hourglass import Hourglass,HourglassFPN,HourglassFPN_2plus1D,HourglassFPN_treble1D

from .cost_processors.utils.cat_fms import CAT_FUNCS
from .cost_processors.utils.dif_fms import DIF_FUNCS
#from .cost_processors.utils.correlation1d_cost import COR_FUNCS
from ..builder import build_cost_aggregator,build_loss

from ..registry import HEADS
from .base_stereo_head import BaseStereoHead

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class LearnableUpsamplingLayer(nn.Module):
    def __init__(self, channel_dim, expansion=64,  ):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_dim, expansion*channel_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(expansion*channel_dim, channel_dim, 3, 1, 1)

    def forward(self, x, target_size):
        B, C, H, W = x.shape
        x_ = F.interpolate(x, size=target_size, mode="bilinear")

        x1 = self.conv1(x)
        x2 = F.interpolate(x1, size=target_size, mode="bilinear")
        x3 = self.conv2(x2)

        out = x3 + x_
        return out

class LearnableUpsamplingLayer3D(nn.Module):
    def __init__(self, channel_dim, expansion=4, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        self.conv1 = nn.Conv3d(channel_dim, channel_dim*expansion, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(channel_dim*expansion, channel_dim,  kernel_size, stride, padding)
    def forward(self, x, target_size):
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        x_ = F.interpolate(x, size=target_size, mode="trilinear")

        x1 = self.conv1(x)
        x2 = F.interpolate(x1, size=target_size, mode="trilinear")
        x3 = self.conv2(x2)

        out = x3 + x_
        return out

class LearnableUpsamplingLayer3Dv2(nn.Module):
    """ rewrite the structure v2, use popular depth-wise+point-wise conv + layernorm, GeLU """
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        self.conv_in = nn.Sequential(   nn.Conv3d(in_channels, latent_channels, 1, 1, 0),
                                        nn.BatchNorm3d(latent_channels),
                                        nn.ReLU(),
                                        nn.Conv3d(latent_channels, latent_channels, 3, 1, 1),
                                        nn.BatchNorm3d(latent_channels),
        )
        self.conv_out = nn.Conv3d(latent_channels, in_channels, 3, 1, 1)
    def forward(self, x, target_size):
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        
        x_ = F.interpolate(x, size=target_size, mode="trilinear",align_corners=False)

        x1 = self.conv_in(x)
        #for i in range(1):
        #    x1 = self.stages[i](x1) + x1
        x2 = F.interpolate(x1, size=target_size, mode="trilinear",align_corners=False)
        x3 = self.conv_out(x2.contiguous()).contiguous()

        out = x3 + x_
        return out

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, D, H, W, C) -> (N, C, D, H, W)

        x = input + x
        return x

class LearnableUpsamplingLayer3Dv3(nn.Module):
    """ keep v2 structure, use LayerNorm and GeLU"""
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        
        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv1 = nn.Sequential(   LayerNorm(in_channels, eps=1e-6),
                                    nn.Linear(in_channels, latent_channels),
                                    nn.GELU(),
        )       
        self.dwconv2 = nn.Sequential( 
            #LayerNorm(latent_channels, eps=1e-6),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=7, padding=3, groups=in_channels ),
        )

    def forward(self, x, target_size):
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        x_ = F.interpolate(x, size=target_size, mode="trilinear",align_corners=False)
        assert C==1
        x = x.squeeze(1) # -> B D H W
        

        x1 = self.dwconv1(x)
        x1 = x1.permute(0,2,3,1)
        x2 = self.pwconv1(x1).permute(0,3,1,2) # -> B 4D H W

        x2 = F.interpolate(x2, size=target_size[-2:], mode="bilinear",align_corners=False) # -> B 4D 4H 4W
        
        x3 = self.dwconv2(x2)#.permute(0,2,3,1)).permute(0,3,1,2) # -> B 4D 4H 4W

        out = x3.unsqueeze(1) + x_ # return B 1 4D 4H 4W
        return out

class LearnableUpsamplingLayer3Dv4(nn.Module):
    """ keep v2 structure, use LayerNorm and GeLU"""
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        
        self.dwconv1 = nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels)
        self.pwconv1 = nn.Sequential(   LayerNorm(in_channels, eps=1e-6),
                                    nn.Linear(in_channels, latent_channels),
                                    nn.GELU(),
        )       
        self.pwconv1_2 =  nn.Linear(latent_channels, latent_channels )

        self.dwconv2 = nn.Conv2d(latent_channels, latent_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv2 = nn.Sequential(   LayerNorm(latent_channels, eps=1e-6),
                                    nn.Linear(latent_channels, latent_channels),
                                    nn.GELU(),
        )
        self.pwconv2_1 = nn.Linear(latent_channels, latent_channels )

    def forward(self, x, target_size):
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        x_ = F.interpolate(x, size=target_size, mode="trilinear",align_corners=False)
        assert C==1
        x = x.squeeze(1) # -> B D H W
        
        x1 = self.dwconv1(x)
        x1 = x1.permute(0,2,3,1)
        x1 = self.pwconv1(x1)
        x2 = self.pwconv1_2(x1)
        x2 = x2.permute(0,3,1,2) # -> B 4D H W
        x2 = F.interpolate(x2, size=target_size[-2:], mode="bilinear",align_corners=False) # -> B 4D 4H 4W
        x3 = self.dwconv2(x2)#.permute(0,2,3,1)).permute(0,3,1,2) # -> B 4D 4H 4W
        x3 = self.pwconv2(x3.permute(0,2,3,1))
        x3 = self.pwconv2_1(x3).permute(0,3,1,2)

        out = x3.unsqueeze(1) + x_ # return B 1 4D 4H 4W
        return out


class LearnableUpsamplingLayer3Dv5(nn.Module):
    """ keep v2 structure, use LayerNorm and GeLU"""
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        
        self.dwconv1 = nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels)
        self.pwconv1 = nn.Sequential(   LayerNorm(in_channels, eps=1e-6),
                                    nn.Linear(in_channels, in_channels*2),
                                    nn.GELU(),
        )       
        self.pwconv1_2 =  nn.Linear(in_channels*2, in_channels*2 )

        self.dwconv2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv2 = nn.Sequential(   LayerNorm(in_channels*2, eps=1e-6),
                                    nn.Linear(in_channels*2, in_channels*2),
                                    nn.GELU(),
        )
        self.pwconv2_1 = nn.Linear(in_channels*2, in_channels*2 )

        self.dwconv3 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv3 = nn.Sequential(   LayerNorm(in_channels*2, eps=1e-6),
                                    nn.Linear(in_channels*2, in_channels*4),
                                    nn.GELU(),
        )
        self.pwconv3_1 = nn.Linear(in_channels*4, in_channels*4 )

        self.dwconv4 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv4 = nn.Sequential(   LayerNorm(in_channels*4, eps=1e-6),
                                    nn.Linear(in_channels*4, in_channels*4),
                                    nn.GELU(),
        )
        self.pwconv4_1 = nn.Linear(in_channels*4, in_channels*4 )

        self.dwconv5 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv5 = nn.Sequential(   LayerNorm(in_channels*4, eps=1e-6),
                                    nn.Linear(in_channels*4, in_channels*4),
        )

     
    def forward(self, x, target_size):
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        res_x4 = F.interpolate(x, scale_factor=4, mode="trilinear",align_corners=False).squeeze(1)
        res_x2 = F.interpolate(x, scale_factor=2, mode="trilinear",align_corners=False).squeeze(1)
        assert C==1
        x = x.squeeze(1) # -> B D H W
        
        x1 = self.dwconv1(x)
        x1 = x1.permute(0,2,3,1)
        x1 = self.pwconv1(x1)
        x2 = self.pwconv1_2(x1)
        x2 = x2.permute(0,3,1,2) # -> B 4D H W
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear",align_corners=False) # -> B 4D 4H 4W
        x3 = self.dwconv2(x2)#.permute(0,2,3,1)).permute(0,3,1,2) # -> B 4D 4H 4W
        x3 = self.pwconv2(x3.permute(0,2,3,1))
        x3 = self.pwconv2_1(x3).permute(0,3,1,2)
        s1 = res_x2 + x3 
        
        s1_x2 = F.interpolate(s1.unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)

        x4 = self.dwconv3(s1)
        x4 = x4.permute(0,2,3,1)
        x4 = self.pwconv3(x4)
        x4 = self.pwconv3_1(x4).permute(0,3,1,2)
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self.dwconv4(x4)
        x5 = x5.permute(0,2,3,1)
        x5 = self.pwconv4(x5)
        x5 = self.pwconv4_1(x5).permute(0,3,1,2)
        s2 = s1_x2 + x5
        x6 = self.dwconv5(s2)
        x6 = self.pwconv5(x6.permute(0,2,3,1)).permute(0,3,1,2)
        x6 = x6 + s2 
        
        s3 = x6 + res_x4

        return s3

class LearnableUpsamplingLayer3Dv6(nn.Module):
    """ keep v2 structure, use LayerNorm and GeLU"""
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        
        self.dwconv1 = nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels)
        self.pwconv1 = nn.Sequential(   LayerNorm(in_channels, eps=1e-6),
                                    nn.Linear(in_channels, in_channels*2),
                                    nn.GELU(),
        )       
        self.pwconv1_2 =  nn.Linear(in_channels*2, in_channels*2 )

        self.dwconv2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv2 = nn.Sequential(   LayerNorm(in_channels*2, eps=1e-6),
                                    nn.Linear(in_channels*2, in_channels*2),
                                    nn.GELU(),
        )
        self.pwconv2_1 = nn.Linear(in_channels*2, in_channels*2 )

        self.dwconv3 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv3 = nn.Sequential(   LayerNorm(in_channels*2, eps=1e-6),
                                    nn.Linear(in_channels*2, in_channels*4),
                                    nn.GELU(),
        )
        self.pwconv3_1 = nn.Linear(in_channels*4, in_channels*4 )

        self.dwconv4 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv4 = nn.Sequential(   LayerNorm(in_channels*4, eps=1e-6),
                                    nn.Linear(in_channels*4, in_channels*4),
                                    nn.GELU(),
        )
        self.pwconv4_1 = nn.Linear(in_channels*4, in_channels*4 )

        self.dwconv5 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=7, padding=3, groups=in_channels)
        self.pwconv5 = nn.Sequential(   LayerNorm(in_channels*4, eps=1e-6),
                                    nn.Linear(in_channels*4, in_channels*4),
        )

     
    def forward(self, x, target_size):
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        #res_x4 = F.interpolate(x, scale_factor=4, mode="trilinear",align_corners=False).squeeze(1)
        res_x2 = F.interpolate(x, scale_factor=2, mode="trilinear",align_corners=False).squeeze(1)
        assert C==1
        x = x.squeeze(1) # -> B D H W
        
        x1 = self.dwconv1(x)
        x1 = x1.permute(0,2,3,1)
        x1 = self.pwconv1(x1)
        x2 = self.pwconv1_2(x1)
        x2 = x2.permute(0,3,1,2) # -> B 4D H W
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear",align_corners=False) # -> B 4D 4H 4W
        x3 = self.dwconv2(x2)#.permute(0,2,3,1)).permute(0,3,1,2) # -> B 4D 4H 4W
        x3 = self.pwconv2(x3.permute(0,2,3,1))
        x3 = self.pwconv2_1(x3).permute(0,3,1,2)
        s1 = res_x2 + x3 
        
        s1_x2 = F.interpolate(s1.unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)

        x4 = self.dwconv3(s1)
        x4 = x4.permute(0,2,3,1)
        x4 = self.pwconv3(x4)
        x4 = self.pwconv3_1(x4).permute(0,3,1,2)
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self.dwconv4(x4)
        x5 = x5.permute(0,2,3,1)
        x5 = self.pwconv4(x5)
        x5 = self.pwconv4_1(x5).permute(0,3,1,2)
        s2 = s1_x2 + x5
        x6 = self.dwconv5(s2)
        x6 = self.pwconv5(x6.permute(0,2,3,1)).permute(0,3,1,2)
        x6 = x6 + s2 
        
        return x6

# Concatenate left and right feature to form cost volume
@HEADS.register_module()
class PSMNetHead48NewCascade(BaseStereoHead):
    def __init__(self, in_channels, disp_range, alpha, normalize, losses=None, grad_method="detach", **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48NewCascade, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
        self.grad_method = grad_method
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

        self.alpha = alpha
        self.normalize = normalize
        self.batch_norm = True
        self._init_layers(self.batch_norm)
        
    def _init_layers(self, batch_norm=True):
        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.in_channels[0], 32, 3, 1, 1, bias=False),
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
        all_pred_disps= []
        pred, cur_disp = None, None
        for stage_idx in range(3):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            if pred is not None:
                if self.grad_method == "detach":
                    cur_disp = pred.detach()
                else:
                    cur_disp = pred
            B,C,H,W = stereo_features[0].shape # 1/4H 
            basebins = self.disp_sample.reshape(1,self.max_disp, 1,1).repeat(B,1,H,W)
            if cur_disp is not None:
                min_cur_disps = torch.clamp(cur_disp - self.max_disp//2, 0, self.max_disp-1)
            else:
                min_cur_disps = torch.zeros_like(basebins).cuda()
            casbins = basebins + min_cur_disps   

            raw_costs = self.cost_builder(stereo_features, casbins)
            decoded_features = self.cost_matcher(raw_costs)
            pred_disps = self.disp_predictor(decoded_features, casbins)
            all_pred_disps = pred_disps + all_pred_disps
            pred = F.interpolate(pred_disps[0], size=(H,W), mode="bilinear")
            
        return all_pred_disps


    def disp_predictor(self, final_costs, casbins):
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

            #assert D == self.disp_sample_number*4, 'The number of disparity samples should be' \
            #                                    ' consistent!'
            #disp_sample = self.disp_sample_pred_layer.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            #disp_sample = disp_sample.to(cost_volume.device)
            disp_sample = 4* F.interpolate(casbins.unsqueeze(1), scale_factor=4, mode="trilinear").squeeze(1)
            # compute disparity: (BatchSize, 1, Height, Width)
            disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)
            
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

    def cost_builder(self, stereo_features, casbins):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        cat_cost = self.fast_cat_fms(ref_fms, tgt_fms, casbins, self.max_disp)
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


    def fast_cat_fms(self, x, y, disp_range_samples, ndisp):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()
        cost = x.new().resize_(bs, channels * 2, ndisp, height, width).zero_()
        # cost = y.unsqueeze(2).repeat(1, 2, ndisp, 1, 1) #(B, 2C, D, H, W)

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = mw - disp_range_samples

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4)   #(B, D, H, W, 2)

        cost[:, x.size()[1]:, :, :, :] = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                                                       padding_mode='zeros').view(bs, channels, ndisp, height, width)

        # a littel difference, no zeros filling
        tmp = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
        # tmp = tmp.transpose(0, 1) #(C, B, D, H, W)
        # #x1 = x2 + d >= d
        # tmp[:, mw < disp_range_samples] = 0
        # tmp = tmp.transpose(0, 1) #(B, C, D, H, W)
        cost[:, :x.size()[1], :, :, :] = tmp

        return cost