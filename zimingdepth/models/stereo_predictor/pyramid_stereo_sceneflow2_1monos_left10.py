'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-22 23:27:35
LastEditors: Ziming Liu
LastEditTime: 2023-03-29 10:47:47
- use classical backbone /4 /8 /16 /32
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import torch.distributed as dist
from mmcv.runner import auto_fp16
from .. import builder
from ..builder import build_head,build_backbone,build_neck

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_deconv, num_filters, deconv_kernels):
        super().__init__()
        self.deconv = num_deconv
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            num_deconv,
            num_filters,
            deconv_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.init_weights()

    def forward(self, conv_feat):
        out = self.deconv_layers(conv_feat)
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

@STEREO_PREDICTOR.register_module()
class PyramidStereoSceneFlow2_1MonosLeft10(nn.Module):
    """
    Base depth method. 

    """
    def __init__(self, backbone, neck, disp_head, mono_head, max_disp, mono_in_channels,
                 stereo_init_head=None, in_channels=128, 
      pretrained=None, mask_size=4, use_adabins=True, search_ranges=[12,24,48], up_sample='linear', output_branch="stereo", prediction_format="left_disps",
      iters=4, **kwargs):
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
        super(PyramidStereoSceneFlow2_1MonosLeft10, self).__init__()
        self.pretrained = pretrained
        self.data = [] # num of frames, each component is a dict{}
        self.iters = iters
        self.max_disp = max_disp
        self.in_channels = in_channels
        self.use_adabins = use_adabins
        self.search_ranges = search_ranges
        self.up_sample = up_sample
        self._MAX_DISP= -1
        self.prediction_format = prediction_format
        self.output_branch = output_branch
        if mono_head is not None:
            self.mono_head = build_head(mono_head)
            self.mono_neck = Decoder(mono_in_channels, mono_head["in_channel"], 
                                    num_deconv=3,
                                    num_filters=[mono_head["in_channel"],mono_head["in_channel"],mono_head["in_channel"],],
                                    deconv_kernels=[2,2,2])
        else:
            self.mono_head = None
              
        if stereo_init_head is not None:
            assert self.mono_head is None
            self.stereo_init_head = build_head(stereo_init_head)
        else:
            self.stereo_init_head = None
        self.disp_head1 = build_head(disp_head[0])
        self.disp_head2 = build_head(disp_head[1])
        self.disp_head3 = build_head(disp_head[2])
        if neck is not None:
            self.neck = build_neck(neck)
        self.neck_cfg = neck
        self.backbone = build_backbone(backbone)
        self.mask_size = mask_size
        self._init_weights_mask()
        self.init_weights()

    def init_weights(self):
        #print([name for name, pa in self.named_parameters()] )
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _init_weights_mask(self, ):
        self.mask0 = nn.Sequential(
            nn.Conv2d(self.in_channels[-1], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4**2 * 9, 1, padding=0),
        )
        self.mask1 = nn.Sequential(
            nn.Conv2d(self.in_channels[-2], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.mask_size**2 * 9, 1, padding=0),
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(self.in_channels[-3], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.mask_size**2 * 9, 1, padding=0),
        )
        self.mask3 = nn.Sequential(
            nn.Conv2d(self.in_channels[-4], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4**2 * 9, 1, padding=0),
        )

  


    def convex_upsample(self, disp, mask, rate=4):
        """[H/rate, W/rate, 2] -> [H, W, 2]"""
        N, _, H, W = disp.shape
        mask = torch.reshape(mask, (N, 1, 9, rate, rate, H, W))
        mask = F.softmax(mask, 2)

        up_disp = F.unfold(rate * disp, [3, 3], padding=1)
        #up_disp = F.reshape(up_disp, (N, 2, 9, 1, 1, H, W))
        up_disp = torch.reshape(up_disp, (N, 1, 9, 1, 1, H, W))

        up_disp = torch.sum(mask * up_disp, 2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return torch.reshape(up_disp, (N, 1, rate * H, rate * W))


    def extract_disp(self, left_img, right_img, **kwargs):
        B,_,H,W = left_img.shape
        imgs = torch.cat([left_img, right_img],0)
        raw_feats = self.backbone(imgs)
        feats = self.neck(raw_feats) # f4 f8 f16 f32 feature size
        
        latent_feats = []
        feats_8 = F.avg_pool2d(feats[0], 2, 2)
        feats_16 = F.avg_pool2d(feats[0], 4, 4)
        latent_feats.append(feats[0])
        latent_feats.append(feats_8)
        latent_feats.append(feats_16)
        
        
        #latent_feats = self.neck(feats)
        out = [] 
        init_out = []
        self.bins_results= []
        if self.stereo_init_head is not None:
            disp_init = self.stereo_init_head([latent_feats[-3][:B,...], latent_feats[-3][B:,...]])
            init_out.extend(disp_init)
            scale = latent_feats[-1].shape[-1] / disp_init[0].shape[-1]
            cur_disps = scale * F.interpolate( disp_init[0], scale_factor=scale, mode="bilinear", align_corners=False)
 
        if self.mono_head is not None:
            mono_feats = self.mono_neck(raw_feats[-1])
            #mask0 = 0.25 * self.mask0(mono_feats[:B,...])
            mono_disp = self.mono_head(mono_feats[:B,...]) # full resolution 
            #disp_convex = self.convex_upsample(mono_disp, mask0, 4)
            #out.append(disp_convex)
            #cur_disps = F.interpolate(1/4 * mono_disp[:B,...], scale_factor=1/4, mode="bilinear", align_corners=False)
            #disp_convex = self.convex_upsample(mono_disp, mask0, 4)
            #out.append(F.interpolate(8*disp_convex, scale_factor=8, mode="bilinear", align_corners=False) )
            #out.append(disp_convex)
            init_out.append(mono_disp)
            #cur_disps = disp_convex
            scale = latent_feats[-1].shape[-1] / mono_disp.shape[-1]
            cur_disps = scale * F.interpolate( mono_disp, scale_factor=scale, mode="bilinear", align_corners=False)
        if self.stereo_init_head is None and self.mono_head is None:
            cur_disps = 0 * torch.ones((B,1,H//16,W//16),device=imgs.device)
        
        # 1/16
        stereo_feat = [latent_feats[-1][:B,...], latent_feats[-1][B:,...]]
        B,_,H,W = cur_disps.shape
        mask1 = 0.25 * self.mask1(latent_feats[-1][:B,...])
        for i in range(self.iters//2):
            cur_disps = torch.relu(cur_disps.detach())
            search_range = self.search_ranges[0]
            #min_cur_disp =  cur_disps - search_range//2 
            #min_cur_disp = torch.clip(min_cur_disp, 0, self.max_disp//16-search_range)
            bins = torch.linspace(-search_range//2,search_range//2-1,search_range,device=cur_disps.device)
            bins = bins.reshape(1,search_range,1,1).repeat(B,1,H,W)
            bins = bins + cur_disps.repeat(1,search_range,1,1)
            #bins = torch.clip(bins, 0, self.max_disp//16)
            delta_cur_disps_list = self.disp_head1(stereo_feat, bins.detach(), )
            assert len(delta_cur_disps_list)==1
            cur_disps = cur_disps + delta_cur_disps_list[0]
            disp_convex = self.convex_upsample(cur_disps, mask1, self.mask_size)
            out.append(F.interpolate(4 * disp_convex, scale_factor=4, mode="bilinear"))
        
        scale = latent_feats[-2].shape[-1] / disp_convex.shape[-1]
        cur_disps = scale * F.interpolate(disp_convex, scale_factor=scale, mode="bilinear", align_corners=False)
        #left_disps_1_x2 = out[-1]
        # 1/8 
        stereo_feat = [latent_feats[-2][:B,...], latent_feats[-2][B:,...]]
        #adabins_feat = F.interpolate(adabins_feat.unsqueeze(1), size=(self.search_ranges[1],)+cur_disps.shape[-2:], mode="trilinear",align_corners=False).squeeze(1)
        B,_,H,W = cur_disps.shape
        mask2 = 0.25 * self.mask2(latent_feats[-2][:B,...])
        for i in range(self.iters//2):
            cur_disps = torch.relu(cur_disps.detach())
            search_range = self.search_ranges[1]
            #min_cur_disp =  cur_disps - search_range//2 
            #min_cur_disp = torch.clip(min_cur_disp, 0, self.max_disp//8-search_range)
            bins = torch.linspace(-search_range//2,search_range//2-1,search_range,device=cur_disps.device)
            bins = bins.reshape(1,search_range,1,1).repeat(B,1,H,W)
            bins = bins + cur_disps.repeat(1,search_range,1,1) 
            #bins = torch.clip(bins, 0, self.max_disp//8)
            delta_cur_disps_list = self.disp_head2(stereo_feat, bins.detach(), )
            assert len(delta_cur_disps_list)==1
            cur_disps = cur_disps + delta_cur_disps_list[0] 
            
            disp_convex = self.convex_upsample(cur_disps, mask2, self.mask_size)
            out.append(F.interpolate(2*disp_convex, scale_factor=2, mode="bilinear"))
        scale = latent_feats[-3].shape[-1] / disp_convex.shape[-1]
        cur_disps = scale * F.interpolate(disp_convex, scale_factor=scale, mode="bilinear", align_corners=False)
        # 1/4 
        stereo_feat = [latent_feats[-3][:B,...], latent_feats[-3][B:,...]]
        #adabins_feat = F.interpolate(adabins_feat.unsqueeze(1), size=(self.search_ranges[2],)+cur_disps.shape[-2:], mode="trilinear",align_corners=False).squeeze(1)
        B,_,H,W = cur_disps.shape
        mask3 = 0.25 * self.mask3(latent_feats[-3][:B,...])
        for i in range(self.iters):
            cur_disps = torch.relu(cur_disps.detach())
            search_range = self.search_ranges[2]
            #min_cur_disp =  cur_disps - search_range//2 
            #min_cur_disp = torch.clip(min_cur_disp, 0, self.max_disp//4-search_range)
            bins = torch.linspace(-search_range//2,search_range//2-1,search_range,device=cur_disps.device)
            bins = bins.reshape(1,search_range,1,1).repeat(B,1,H,W)
            bins = bins + cur_disps.repeat(1,search_range,1,1) 
            #bins = torch.clip(bins, 0, self.max_disp//4)
            delta_cur_disps_list = self.disp_head3(stereo_feat, bins.detach())
            assert len(delta_cur_disps_list)==1
            cur_disps = cur_disps + delta_cur_disps_list[0]
            
            out.append(self.convex_upsample(cur_disps, mask3, 4))
        
        return out,init_out
    
    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        B = left_imgs.shape[0]
        out,init_out = self.extract_disp(left_imgs, right_imgs,  **kwargs)
        if self.mono_head is not None:
            mono_disp = init_out
            stereo_disp = out
            # mono loss
            #gt_depth = (kwargs["focal"]*kwargs["baseline"]).reshape(B,1,1,1) / (kwargs["left_disps"] + 1e-6)
            loss_mono = self.mono_head.loss(mono_disp,  kwargs["left_disps"]  )
            losses.update(loss_mono)
            # stereo loss
            loss_stereo = self.disp_head1.loss(stereo_disp, kwargs["left_disps"])
            losses.update(loss_stereo)
        else:
            stereo_init = init_out
            stereo_disp = out
            loss_init = self.stereo_init_head.loss(stereo_init, kwargs["left_disps"])
            losses.update(loss_init)
            loss_stereo = self.disp_head1.loss(stereo_disp, kwargs["left_disps"])
            losses.update(loss_stereo)
  
        return losses
 
 
    def forward_test(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        frame_dir = None

        return self._do_test( left_imgs, right_imgs, frame_dir,**kwargs)
        
    def _do_test(self, left_imgs, right_imgs,frame_dir,**kwargs):
        """Defines the computation performed at every call when training."""
        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir
        B = left_imgs.shape[0]
        out,init_out = self.extract_disp(left_imgs, right_imgs,  **kwargs)
        stereo_disp = out
        if self.output_branch == "stereo":
            outputs[0] = stereo_disp[-1].detach().cpu().numpy()
        else:
            outputs[0] = init_out[0][:B,...].detach().cpu().numpy()
        if "left_disps" in kwargs:
            outputs[1] = kwargs["left_disps"].squeeze(1).unsqueeze(1).detach().cpu().numpy()
        else:
            outputs[1] = []
        return outputs
     
    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items())
                #   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, left_imgs, right_imgs =None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        #assert "left_disps" in kwargs and "right_disps" in kwargs
        if return_loss:
            return self.forward_train(left_imgs, right_imgs, **kwargs)

        return self.forward_test(left_imgs, right_imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        left_imgs, right_imgs = data_batch['left_imgs'], data_batch['right_imgs']
        #label = data_batch['pose']
        
        aux_info = {}
        # if aux info is not define in config, there 
        #for item in self.aux_info:
        #    assert item in data_batch
        #    aux_info[item] = data_batch[item]
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        keys.remove('right_imgs')
        #keys.remove('pose')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
        # print gradient of network
        #for name, parms in self.named_parameters():
        #    print('-->name:', name, '-->grad_requirs:',parms.requires_grad,  )
        #    print(' -->grad_value: \n {}'.format(parms.grad))

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        left_imgs, right_imgs = data_batch['left_imgs'], data_batch['right_imgs']
        #label = data_batch['pose']

        aux_info = {}
        #for item in self.aux_info:
        #    aux_info[item] = data_batch[item]
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        keys.remove('right_imgs')
        #keys.remove('pose')
        for item in keys:
            aux_info[item] = data_batch[item]

            
        outputs = self(left_imgs, right_imgs, return_loss=False, **aux_info)


        return outputs



