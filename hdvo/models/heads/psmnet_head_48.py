"""
PSMNet-based Stereo Matching Head.

This module implements various PSMNet-based stereo matching heads with
different cost aggregation and upsampling strategies.

Reference: https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark

Author: Ziming Liu
Date: 2022-07-07
Last Modified: 2024-02-08
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

class LayerNorm(nn.Module):
    """Layer Normalization supporting multiple data formats.
    
    Supports both channels_last (NHWC) and channels_first (NCHW) formats.
    
    Args:
        normalized_shape (int): Number of features to normalize.
        eps (float): Small value to avoid division by zero. Defaults to 1e-6.
        data_format (str): Either 'channels_last' or 'channels_first'.
            Defaults to 'channels_last'.
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
        """Apply layer normalization.
        
        Args:
            x: Input tensor.
            
        Returns:
            Normalized tensor.
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class LearnableUpsamplingLayer(nn.Module):
    """2D learnable upsampling layer with residual connection.
    
    Args:
        channel_dim (int): Number of input/output channels.
        expansion (int): Channel expansion factor. Defaults to 64.
    """
    
    def __init__(self, channel_dim, expansion=64,  ):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_dim, expansion*channel_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(expansion*channel_dim, channel_dim, 3, 1, 1)

    def forward(self, x, target_size):
        """Forward pass with learnable upsampling.
        
        Args:
            x: Input tensor [B, C, H, W].
            target_size: Target spatial size (H', W').
            
        Returns:
            Upsampled tensor [B, C, H', W'].
        """
        B, C, H, W = x.shape
        x_ = F.interpolate(x, size=target_size, mode="bilinear")

        x1 = self.conv1(x)
        x2 = F.interpolate(x1, size=target_size, mode="bilinear")
        x3 = self.conv2(x2)

        out = x3 + x_
        return out

class LearnableUpsamplingLayer3D(nn.Module):
    """3D learnable upsampling layer with residual connection.
    
    Args:
        channel_dim (int): Number of input/output channels.
        expansion (int): Channel expansion factor. Defaults to 4.
        kernel_size (int): Convolution kernel size. Defaults to 3.
        stride (int): Convolution stride. Defaults to 1.
        padding (int): Convolution padding. Defaults to 1.
    """
    
    def __init__(self, channel_dim, expansion=4, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        self.conv1 = nn.Conv3d(channel_dim, channel_dim*expansion, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(channel_dim*expansion, channel_dim,  kernel_size, stride, padding)
        
    def forward(self, x, target_size):
        """Forward pass with 3D learnable upsampling.
        
        Args:
            x: Input tensor [B, C, D, H, W].
            target_size: Target size (D', H', W').
            
        Returns:
            Upsampled tensor [B, C, D', H', W'].
        """
        B, C, D, H, W = x.shape
        assert len(target_size)==3
        x_ = F.interpolate(x, size=target_size, mode="trilinear")

        x1 = self.conv1(x)
        x2 = F.interpolate(x1, size=target_size, mode="trilinear")
        x3 = self.conv2(x2)

        out = x3 + x_
        return out

class LearnableUpsamplingLayer3Dv2(nn.Module):
    """Improved 3D learnable upsampling with depth-wise and point-wise convolutions.
    
    Uses depth-wise + point-wise convolution with batch normalization.
    
    Args:
        in_channels (int): Number of input channels.
        latent_channels (int): Number of latent channels.
        kernel_size (int): Convolution kernel size. Defaults to 3.
        stride (int): Convolution stride. Defaults to 1.
        padding (int): Convolution padding. Defaults to 1.
    """
    
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1 ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, latent_channels, 1, 1, 0),
            nn.BatchNorm3d(latent_channels),
            nn.ReLU(),
            nn.Conv3d(latent_channels, latent_channels, 3, 1, 1),
            nn.BatchNorm3d(latent_channels),
        )
        self.conv_out = nn.Conv3d(latent_channels, in_channels, 3, 1, 1)
        
    def forward(self, x, target_size):
        """Forward pass with improved 3D upsampling.
        
        Args:
            x: Input tensor [B, C, D, H, W].
            target_size: Target size (D', H', W').
            
        Returns:
            Upsampled tensor [B, C, D', H', W'].
        """
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
class PSMNetHead48(BaseStereoHead):
    def __init__(self, in_channels, disp_range, alpha, normalize, losses=None,
                 fast_cat=False, local_predictor=False, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
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
        self.fast_cat = fast_cat
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

    def cost_builder(self, stereo_features):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        if not self.fast_cat:
            cat_cost = self.cat_fms(ref_fms, tgt_fms,)
        else:
            cat_cost = self.fast_cat_fms(ref_fms, tgt_fms)
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


    def fast_cat_fms(self, reference_fm, target_fm):
        device = reference_fm.device
        B, C, H, W = reference_fm.shape
        D = self.disp_sample_number
        disp_sample = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -disp_sample.float(), padding_mode='zeros')

        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm

@HEADS.register_module()
class PSMNetHead48onelevelMono2StereoRemove(nn.Module):
    def __init__(self, in_channels, max_disp, alpha, normalize, losses=None, learn_resize=None,  **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevelMono2StereoRemove, self).__init__()
        self.in_channels = in_channels
        self.learn_resize = learn_resize
 
        self.max_disp = max_disp
         
        self.disp_loss_func = build_loss(losses)
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
        self.dres2 = HourglassFPN(in_planes=32, batch_norm=batch_norm)
 
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if self.learn_resize == "v2":
            self.resize_layer = LearnableUpsamplingLayer3Dv2(1, 16)
        if self.learn_resize == "v3":
            self.resize_layer = LearnableUpsamplingLayer3Dv3(self.disp_sample_number, 4*self.disp_sample_number)
        if self.learn_resize == "v4":
            self.resize_layer = LearnableUpsamplingLayer3Dv4(self.disp_sample_number, 4*self.disp_sample_number)
        if self.learn_resize == "v5":
            self.resize_layer = LearnableUpsamplingLayer3Dv5(self.disp_sample_number, self.disp_sample_number*4)
        if self.learn_resize == "v6":
            self.resize_layer = LearnableUpsamplingLayer3Dv6(self.disp_sample_number, self.disp_sample_number*4)

    def forward(self, feats, mono_disp):
        if isinstance(feats[0], (list,tuple)):
            B,C,H,W = feats[0][0].shape
        else:
            B,C,H,W = feats[0].shape
        bins = torch.linspace(-self.max_disp//8,self.max_disp//8-1,self.max_disp//4).reshape(1,self.max_disp//4,1,1).repeat(B,1,H,W).cuda()
        bins = bins + (1/4)*F.interpolate(mono_disp.detach().clone(), size=(H,W), mode="bilinear", align_corners=False).repeat(1,self.max_disp//4,1,1)
        bins = torch.relu(bins)
        costs = self.cost_builder(feats,bins)
        costs = self.cost_matcher(costs)
        #newbins = 4* F.interpolate(bins.unsqueeze(1), size=(self.max_disp,4*H,4*W),mode="trilinear",align_corners=False).squeeze(1)
        wider_bins = torch.linspace(-self.max_disp//2,self.max_disp//2-1,self.max_disp).reshape(1,self.max_disp,1,1).repeat(B,1,4*H,4*W).cuda()
        wider_bins = wider_bins + mono_disp.repeat(1,self.max_disp,1,1)
        
        disp = self.disp_predictor(costs, wider_bins)
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
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0
 
        cost1 = self.classif1(out1).contiguous()
        if self.learn_resize is None:
            cost1 = F.interpolate(
                cost1, [self.max_disp, H * 4, W * 4],
                mode='trilinear', align_corners=False
            )
        elif self.learn_resize is not None:
            cost1 = self.resize_layer(cost1, [self.max_disp, H * 4, W * 4])
        else:
            raise ValueError

        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
 
        return ( cost1)

    def cost_builder(self, stereo_features,bins):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        #print(ref_fms.shape)
        cat_cost = self.cat_fms(ref_fms, tgt_fms,bins)
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
        
    def fast_cat_fms(self, reference_fm, target_fm, bins):
        device = reference_fm.device
        B, C, H, W = reference_fm.shape
        D = self.max_disp//4
        #disp_sample = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -bins.float(), padding_mode='zeros')

        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm

    def loss(self, pred, gt, **kwargs):
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses


@HEADS.register_module()
class PSMNetHead48onelevel(BaseStereoHead):
    def __init__(self, in_channels, disp_range, alpha, normalize, losses=None, learn_resize=None,  **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevel, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
        self.learn_resize = learn_resize
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
 
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if self.learn_resize == "v2":
            self.resize_layer = LearnableUpsamplingLayer3Dv2(1, 16)
        if self.learn_resize == "v3":
            self.resize_layer = LearnableUpsamplingLayer3Dv3(self.disp_sample_number, 4*self.disp_sample_number)
        if self.learn_resize == "v4":
            self.resize_layer = LearnableUpsamplingLayer3Dv4(self.disp_sample_number, 4*self.disp_sample_number)
        if self.learn_resize == "v5":
            self.resize_layer = LearnableUpsamplingLayer3Dv5(self.disp_sample_number, self.disp_sample_number*4)
        if self.learn_resize == "v6":
            self.resize_layer = LearnableUpsamplingLayer3Dv6(self.disp_sample_number, self.disp_sample_number*4)


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
 
        cost1 = self.classif1(out1).contiguous()
        if self.learn_resize is None:
            cost1 = F.interpolate(
                cost1, [self.max_disp*4, H * 4, W * 4],
                mode='trilinear', align_corners=False
            )
        elif self.learn_resize is not None:
            cost1 = self.resize_layer(cost1, [self.max_disp*4, H * 4, W * 4])
        else:
            raise ValueError

        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
 
        return ( cost1)

    def cost_builder(self, stereo_features):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        #print(ref_fms.shape)
        cat_cost = self.cat_fms(ref_fms, tgt_fms,)
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


    def fast_cat_fms(self, reference_fm, target_fm):
        device = reference_fm.device
        B, C, H, W = reference_fm.shape
        D = self.disp_sample_number
        disp_sample = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -disp_sample.float(), padding_mode='zeros')

        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm


@HEADS.register_module()
class PSMNetHead48onelevelFPN(PSMNetHead48onelevel):
    def __init__(self, in_channels, disp_range, alpha, normalize, losses=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevelFPN, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
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
        self.dres2 = HourglassFPN(in_planes=32, batch_norm=batch_norm)
 
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

@HEADS.register_module()
class PSMNetHead48onelevelFPNtreble1D(PSMNetHead48onelevel):
    def __init__(self, in_channels, disp_range, alpha, normalize, losses=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevelFPNtreble1D, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
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
            conv3d_bn_relu(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,3,1), 1, (0,1,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,1,3), 1, (0,0,1), bias=False),
        )
        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,3,1), 1, (0,1,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,1,3), 1, (0,0,1), bias=False),
            conv3d_bn(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn(batch_norm, 32, 32, (1,3,1), 1, (0,1,0), bias=False),
            conv3d_bn(batch_norm, 32, 32, (1,1,3), 1, (0,0,1), bias=False),
            
        )
        self.dres2 = HourglassFPN_treble1D(in_planes=32, batch_norm=batch_norm)
 
        self.classif1 = nn.Sequential(
            #conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,3,1), 1, (0,1,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,1,3), 1, (0,0,1), bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

@HEADS.register_module()
class PSMNetHead48onelevelFPN2plus1D(PSMNetHead48onelevel):
    def __init__(self, in_channels, disp_range, alpha, normalize, losses=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevelFPN2plus1D, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
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
            #conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,3,3), 1, (0,1,1), bias=False),
        )
        self.dres1 = nn.Sequential(
            #conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,3,3), 1, (0,1,1), bias=False),
            conv3d_bn(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn(batch_norm, 32, 32, (1,3,3), 1, (0,1,1), bias=False),
            #conv3d_bn(batch_norm, 32, 32, 3, 1, 1, bias=False)
        )
        self.dres2 = HourglassFPN_2plus1D(in_planes=32, batch_norm=batch_norm)
 
        self.classif1 = nn.Sequential(
            #conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (3,1,1), 1, (1,0,0), bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, (1,3,3), 1, (0,1,1), bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

@HEADS.register_module()
class PSMNetHead48onelevelPostResize(BaseStereoHead):
    def __init__(self, in_channels, disp_range, alpha, normalize, learn_resize=False,  losses=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevelPostResize, self).__init__(in_channels, disp_range, alpha, normalize,  losses, **kwargs)
        self.in_channels = in_channels
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
        self.learn_resize = learn_resize
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
 
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        if self.learn_resize:
            self.post_resize = LearnableUpsamplingLayer(1,128)

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
 
        cost1 = self.classif1(out1)
 
        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
 
        return ( cost1)

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

            disp_sample = self.disp_sample.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            disp_sample = disp_sample.to(cost_volume.device)

        
            # compute disparity: (BatchSize, 1, Height, Width)
            disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)
            
            pred_disps.append(disp_map)

        return pred_disps

    def forward(self, stereo_features, disp_sample=None):
        raw_costs = self.cost_builder(stereo_features)
        decoded_features = self.cost_matcher(raw_costs)
        pred_disps = self.disp_predictor(decoded_features)
        for i in range(len(pred_disps)):
            B, C, H, W = pred_disps[i].shape
            if self.learn_resize:
                pred_disps[i] = 4 * self.post_resize(pred_disps[i], (4*H, 4*W))
            else:
                pred_disps[i] = 4 * F.interpolate(pred_disps[i], size=(4*H, 4*W), mode="bilinear" )
            
        return pred_disps

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


    def fast_cat_fms(self, reference_fm, target_fm):
        device = reference_fm.device
        B, C, H, W = reference_fm.shape
        D = self.disp_sample_number
        disp_sample = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -disp_sample.float(), padding_mode='zeros')

        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm


@HEADS.register_module()
class PSMNetHead48onelevelv2(BaseStereoHead):
    def __init__(self, in_channels, disp_range, alpha, normalize, FPNhourglass=False, adabins="pre-define", learn_resize=None, losses=None, **kwargs):
        '''
        description: 
        return: {*}
        '''        
        super(PSMNetHead48onelevelv2, self).__init__(in_channels, disp_range, alpha, normalize, losses, **kwargs)
        self.in_channels = in_channels
        self.learn_resize = learn_resize
        self.adabins = adabins
        self.FPNhourglass = FPNhourglass
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
        if self.FPNhourglass:
            self.dres2 = HourglassFPN(in_planes=32, batch_norm=batch_norm)
        else:
            self.dres2 = Hourglass(in_planes=32, batch_norm=batch_norm)
 
        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if self.adabins == "pixel-wise":
            self.bins_transform = nn.Sequential(
                conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            )
        if self.adabins == "pixel-wise" and self.learn_resize == "v5":
            self.resize_layer_adabins = LearnableUpsamplingLayer3Dv5(self.disp_sample_number, self.disp_sample_number*4)

        if self.learn_resize == "v2":
            self.resize_layer = LearnableUpsamplingLayer3Dv2(1, 16)
        if self.learn_resize == "v3":
            self.resize_layer = LearnableUpsamplingLayer3Dv3(self.disp_sample_number, 4*self.disp_sample_number)
        if self.learn_resize == "v4":
            self.resize_layer = LearnableUpsamplingLayer3Dv4(self.disp_sample_number, 4*self.disp_sample_number)
        if self.learn_resize == "v5":
            self.resize_layer = LearnableUpsamplingLayer3Dv5(self.disp_sample_number, self.disp_sample_number*4)
        if self.learn_resize == "v6":
            self.resize_layer = LearnableUpsamplingLayer3Dv6(self.disp_sample_number, self.disp_sample_number*4)


    def forward(self, stereo_features, disp_sample=None):
        left_feat = stereo_features[0][0]
        B,C,H,W = left_feat.shape
        raw_costs = self.cost_builder(stereo_features)
        decoded_features, adabins = self.cost_matcher(raw_costs)
        if self.adabins == "pre-define":
            bins = self.disp_sample_pred_layer.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            bins = bins.cuda()
        if self.adabins == "instance-wise" or self.adabins == "pixel-wise":
            binH, binW = adabins.shape[-2:]
            ada_dist = F.softmax(adabins,1).reshape((B,self.disp_sample_number*4,binH*binW)).permute(0,2,1)
            bin_widths_normed = ada_dist / (ada_dist.sum(-1,True) + 1e-6)
            min_disp = 0 
            bin_widths = (self.max_disp*4 - min_disp) * bin_widths_normed  # B HW D
            bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=min_disp)
            bin_edges = torch.cumsum(bin_widths, dim=-1) # accumulated sum
            centers = 0.5 * (bin_edges[:, :, :-1] + bin_edges[:, :, 1:])
            bins = centers
            bins = bins.permute(0,2,1).reshape((B,self.disp_sample_number*4, binH, binW) )
            bins = bins.cuda()
        pred_disps = self.disp_predictor(decoded_features, bins)

        return pred_disps


    def disp_predictor(self, final_costs, disp_sample=None):
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

            if disp_sample is None:
                disp_sample = self.disp_sample_pred_layer.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
                disp_sample = disp_sample.to(cost_volume.device)                
        
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
 
        cost1 = self.classif1(out1).contiguous()
        adabins = None
        if self.adabins == "pixel-wise":
            adabins = self.bins_transform(out1)
            if self.learn_resize == "v5":
                adabins = self.resize_layer_adabins(adabins, [self.max_disp*4, H*4, W*4])
            else:
                adabins = F.interpolate(adabins,[self.max_disp*4, H * 4, W * 4],
                    mode='trilinear', align_corners=False)
            
            adabins = adabins.squeeze(1)

        if self.learn_resize is None:
            cost1 = F.interpolate(
                cost1, [self.max_disp*4, H * 4, W * 4],
                mode='trilinear', align_corners=False
            )
        elif self.learn_resize is not None:
            cost1 = self.resize_layer(cost1, [self.max_disp*4, H * 4, W * 4])
        else:
            raise ValueError

        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
 
        return ( cost1), adabins

    def cost_builder(self, stereo_features):
        assert len(stereo_features) == 2, "stereo inputs"
        if isinstance(stereo_features[0], list) or isinstance(stereo_features[0], tuple):
            #assert len(stereo_features[0]) == 1
            stereo_features[0],stereo_features[1] = stereo_features[0][0], stereo_features[1][0]
        ref_fms, tgt_fms = stereo_features[0], stereo_features[1]
        #print(ref_fms.shape)
        cat_cost = self.cat_fms(ref_fms, tgt_fms,)
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
    def fast_cat_fms(self, reference_fm, target_fm):
        device = reference_fm.device
        B, C, H, W = reference_fm.shape
        D = self.disp_sample_number
        disp_sample = self.disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W).to(device).type_as(reference_fm)
        # expand D dimension
        concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

        # shift target feature according to disparity samples
        concat_target_fm = inverse_warp_3d(concat_target_fm.float(), -disp_sample.float(), padding_mode='zeros')

        # mask out features in reference
        concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(reference_fm) # fix the type bug when using half-float. ziming 21-7-8

        # [B, 2C, D, H, W)
        concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

        return concat_fm