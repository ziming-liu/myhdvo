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

import torch.distributed as dist
from mmcv.runner import auto_fp16
from .. import builder
from ..builder import build_head,build_backbone,build_neck

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor

@STEREO_PREDICTOR.register_module()
class PyramidStereoSceneFlow2_1(nn.Module):
    """
    Base depth method. 

    """
    def __init__(self, backbone, neck, disp_head, mono_head, in_channels=128,
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
        super(PyramidStereoSceneFlow2_1, self).__init__()
        self.pretrained = pretrained
        self.data = [] # num of frames, each component is a dict{}
        self.iters = iters
        self.in_channels = in_channels
        self.use_adabins = use_adabins
        self.search_ranges = search_ranges
        self.up_sample = up_sample
        self._MAX_DISP= -1
        self.prediction_format = prediction_format
        self.output_branch = output_branch
        self.mono_head = build_head(mono_head)      
        self.disp_head1 = build_head(disp_head)
        self.disp_head2 = build_head(disp_head)
        self.disp_head3 = build_head(disp_head)
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
        #self.mask0 = nn.Sequential(
        #    nn.Conv2d(self.in_channels, 128, 3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(128, self.mask_size**2 * 9, 1, padding=0),
        #)
        self.mask1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.mask_size**2 * 9, 1, padding=0),
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.mask_size**2 * 9, 1, padding=0),
        )
        self.mask3 = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4**2 * 9, 1, padding=0),
        )

        # adaptive search
        

        # adaptive bins 
        

    def adabins(self, adabins, min_disp):
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
        feats = self.backbone(imgs)
        latent_feats = []
        feats_8 = F.avg_pool2d(feats[-1], 2, 2)
        feats_16 = F.avg_pool2d(feats_8, 2, 2)
        latent_feats.append(feats[-1])
        latent_feats.append(feats_8)
        latent_feats.append(feats_16)
        #latent_feats = self.neck(feats)
        out = [] 
        self.bins_results= []
        mono_disp = self.mono_head(latent_feats[-1]) # 1/16 resolution 
        #mask0 = self.mask0(latent_feats[-1])
        #mono_disp_x4= self.convex_upsample(mono_disp, mask0, 4)
        #mono_disp_x4 = F.interpolate(4*mono_disp_x4, scale_factor=4, mode="bilinear", align_corners=False) 
        out.append(mono_disp)
        # 1/16
        stereo_feat = [latent_feats[-1][:B,...], latent_feats[-1][B:,...]]
        cur_disps = mono_disp[:B,...]
        adabins_feat= None
        B,_,H,W = cur_disps.shape
        for i in range(self.iters//2):
            search_range = self.search_ranges[0]
            min_cur_disp = torch.clip(cur_disps - search_range//2, 0)
            #max_cur_disp = cur_disps + search_range//2
            #if self.use_adabins and adabins_feat is not None:
                #adabins_feat = self.conv_adabins1(latent_feats[-1][:B,...])
            #    bins = self.adabins(adabins_feat, min_cur_disp.detach())
                #bins = torch.clip(bins, 0, )
                #print(bins[0,:,10,10])
            #    self.bins_results.append(bins)
            #else:
            bins = torch.linspace(0,search_range-1,search_range,device=cur_disps.device)
            bins = bins.reshape(1,search_range,1,1).repeat(B,1,H,W)
            bins = bins + min_cur_disp.repeat(1,search_range,1,1).detach()
            #bins = torch.clip(bins, 0, )
            #offset_dw16 = self.conv_offset_16(latent_feats[-1][:B,...])
            #y_bins = self.range_16 * (torch.sigmoid(offset_dw16) - 0.5) * 2.0
            #bins = torch.clip(bins, 0, search_range-1)
            cur_disps_list = self.disp_head1(stereo_feat, bins, )
            #cur_disps = cur_disps[0] if isinstance(cur_disps,(list,tuple)) else cur_disps
            mask1 = 0.25 * self.mask1(latent_feats[-1][:B,...])
            #left_disps_1_x4 = self.convex_upsample(cur_disps, mask1, 4)
            #left_disps_1_x4x4 = F.interpolate(4*left_disps_1_x4, scale_factor=4, mode="bilinear", align_corners=False) 
            for cur_disps in cur_disps_list:
                out.append(self.convex_upsample(cur_disps, mask1, 2))
        #left_disps_1_x2 = F.interpolate(cur_disps * 2, scale_factor=2, mode="bilinear", align_corners=False)
        left_disps_1_x2 = out[-1]
        # 1/8 
        stereo_feat = [latent_feats[-2][:B,...], latent_feats[-2][B:,...]]
        cur_disps = left_disps_1_x2
        #adabins_feat = F.interpolate(adabins_feat.unsqueeze(1), size=(self.search_ranges[1],)+cur_disps.shape[-2:], mode="trilinear",align_corners=False).squeeze(1)
        B,_,H,W = cur_disps.shape
        for i in range(self.iters//2):
            search_range = self.search_ranges[1]
            min_cur_disp = torch.clip(cur_disps - search_range//2, 0)
            #max_cur_disp = cur_disps + search_range//2
            #if self.use_adabins and adabins_feat is not None:
                #adabins_feat = self.conv_adabins2(latent_feats[-2][:B,...])
            #    bins = self.adabins(adabins_feat, min_cur_disp.detach())
                #bins = torch.clip(bins, 0, )
                #print(bins[0,:,20,20])
            #    self.bins_results.append(bins)
            #else:
            bins = torch.linspace(0,search_range-1,search_range,device=cur_disps.device)
            bins = bins.reshape(1,search_range,1,1).repeat(B,1,H,W)
            bins = bins + min_cur_disp.repeat(1,search_range,1,1).detach()
            #bins = torch.clip(bins, 0, )
            #offset_dw8 = self.conv_offset_8(latent_feats[-2][:B,...])
            #y_bins = self.range_8 * (torch.sigmoid(offset_dw8) - 0.5) * 2.0
            #bins = torch.clip(bins, 0, search_range-1)
            cur_disps_list = self.disp_head2(stereo_feat, bins.detach(), )
            #cur_disps = cur_disps[0] if isinstance(cur_disps,(list,tuple)) else cur_disps
            mask2 = 0.25 * self.mask2(latent_feats[-2][:B,...])
            #left_disps_2_x4 = self.convex_upsample(cur_disps, mask2, 4)
            #left_disps_2_x4x2 = F.interpolate(2*left_disps_2_x4, scale_factor=2, mode="bilinear", align_corners=False) 
            for cur_disps in cur_disps_list:
                out.append(self.convex_upsample(cur_disps, mask2, 2))
        #left_disps_2_x2 = F.interpolate(cur_disps * 2, scale_factor=2, mode="bilinear", align_corners=False)
        left_disps_2_x2 = out[-1]
        # 1/4 
        stereo_feat = [latent_feats[-3][:B,...], latent_feats[-3][B:,...]]
        cur_disps = left_disps_2_x2
        #adabins_feat = F.interpolate(adabins_feat.unsqueeze(1), size=(self.search_ranges[2],)+cur_disps.shape[-2:], mode="trilinear",align_corners=False).squeeze(1)
        B,_,H,W = cur_disps.shape
        for i in range(self.iters):
            search_range = self.search_ranges[2]
            min_cur_disp = torch.clip(cur_disps - search_range//2, 0)
            #max_cur_disp = cur_disps + search_range//2
            #if self.use_adabins and adabins_feat is not None:
                #adabins_feat = self.conv_adabins3(latent_feats[-3][:B,...])
            #    bins = self.adabins(adabins_feat, min_cur_disp.detach())
                #bins = torch.clip(bins, 0, )
                #print(bins[0,:,40,40])
            #    self.bins_results.append(bins)
            #else:
            bins = torch.linspace(0,search_range-1,search_range,device=cur_disps.device)
            bins = bins.reshape(1,search_range,1,1).repeat(B,1,H,W)
            bins = bins + min_cur_disp.repeat(1,search_range,1,1).detach()
            #bins = torch.clip(bins, 0, )
            #bins = torch.clip(bins, 0, search_range-1)
            cur_disps_list = self.disp_head3(stereo_feat, bins.detach())
            #cur_disps = cur_disps[0] if isinstance(cur_disps,(list,tuple)) else cur_disps
            mask3 = 0.25 * self.mask3(latent_feats[-3][:B,...])
            #left_disps_3_x4 = F.interpolate(left_disps_3, scale_factor=4, mode="bilinear", align_corners=False)
            for cur_disps in cur_disps_list:
                out.append(self.convex_upsample(cur_disps, mask3, 4))        
        
        return out


    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        B = left_imgs.shape[0]
        out = self.extract_disp(left_imgs, right_imgs,  **kwargs)
        mono_disp = out[0]
        stereo_disp = out[1:]
        # mono loss
        #gt_depth = (kwargs["focal"]*kwargs["baseline"]).reshape(B,1,1,1) / (kwargs["left_disps"] + 1e-6)
        #loss_mono = self.mono_head.loss(mono_disp, torch.cat([kwargs["left_disps"], kwargs["right_disps"]],0) )
        loss_mono = self.mono_head.loss(mono_disp, torch.cat([kwargs["left_disps"], kwargs["right_disps"]],0) )
        
        losses.update(loss_mono)
        # stereo loss
        flow_losses = dict()
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
        out = self.extract_disp(left_imgs, right_imgs,  **kwargs)
        mono_disp = out[0]
        stereo_disp = out[-1]
        if self.output_branch == "stereo":
            outputs[0] = stereo_disp.squeeze(1).unsqueeze(1).detach().cpu().numpy()
        else:
            outputs[0] = mono_disp[:B,...].squeeze(1).unsqueeze(1).detach().cpu().numpy()
        outputs[1] = torch.stack( [kwargs["left_disps"].squeeze(1), 
                    kwargs["right_disps"].squeeze(1) ], 1).detach().cpu().numpy()
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
        assert "left_disps" in kwargs and "right_disps" in kwargs
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



@STEREO_PREDICTOR.register_module()
class PyramidStereoSceneFlow2_1Unsup(PyramidStereoSceneFlow2_1):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, unsup_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", TTT=False, **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)

        self.unsup_head = build_head(unsup_head)
        self.TTT = TTT


    def forward_train(self, left_imgs, right_imgs=None, **kwargs):
        losses = dict()
        B = left_imgs.shape[0]
        mono_disp, stereo_disp = self.extract_disp(left_imgs, right_imgs,  **kwargs)

        # mono loss
        if not self.TTT:
            #gt_depth = (kwargs["focal"]*kwargs["baseline"]).reshape(B,1,1,1) / (kwargs["left_disps"] + 1e-6)
            loss_mono = self.mono_head.loss(mono_disp, kwargs["left_disps"] )
            losses.update(loss_mono)
            # stereo loss
            loss_stereo = self.disp_head.loss(stereo_disp, kwargs["left_disps"])
            losses.update(loss_stereo)

        # unsupervised optimization 
        mono_warped_leftimg, mono_warped_rightimg =  self.unsup_head(left_imgs, right_imgs, mono_disp)
        stereo_warped_leftimg, stereo_warped_rightimg = self.unsup_head(left_imgs, right_imgs, stereo_disp)
        loss_unsup_mono = self.unsup_head.loss(mono_warped_leftimg, left_imgs, "monoheadL")
        losses.update(loss_unsup_mono)
        loss_unsup_stereo = self.unsup_head.loss(stereo_warped_leftimg, left_imgs, "stereoheadL")
        losses.update(loss_unsup_stereo)

        return losses


@STEREO_PREDICTOR.register_module()
class PyramidStereoSceneFlow2_1v2(PyramidStereoSceneFlow2_1):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)



    def extract_disp(self, left_img, right_img, **kwargs):
        B,_,H,W = left_img.shape
        all = torch.cat([left_img, right_img],0)
        all_feats = self.mono_backbone(all)
        left_feat, right_feat = [all_feats[i][:B,...] for i in range(len(all_feats))],  [all_feats[i][B:,...] for i in range(len(all_feats))]
        #left_feat2 = self.stereo_backbone_left(left_img)
        left_feat2 = [left_feat[1]]
        right_feat2 = [right_feat[1]]
        #right_feat = self.stereo_backbone_right(right_img)
        #if not isinstance(right_feat, (list,tuple)):
        #    right_feat = [right_feat]
        #assert len(left_feat)>1, "backbone gives 4 feats"
        # mono branch 
        mono_feats = [ torch.cat([left_feat[i], right_feat[i]],0) for i in range(1,len(left_feat))]
        mono_feat = self.mono_neck(mono_feats)
        #if len(mono_feat)>1: # FPN outputs ()
        #    mono_feat = mono_feat[0]
        #mono_feat = F.interpolate(mono_feat, size=(H,W), mode="bilinear", align_corners=False)
        mono_disp = [self.mono_head(mono_feat[i]) for i in range(len(mono_feat))]
        #mono_disp = (kwargs['focal']*kwargs['baseline']).reshape(B,1,1,1)  / (1e-6+mono_depth[:B,...])
        #print(left_feat2[1].shape)
        #exit()
        #last_disp = mono_disp[-1]
        for i in range(len(mono_disp)-2, -1, -1):
            mono_disp[i] = mono_disp[i] + F.interpolate(2*mono_disp[i+1],scale_factor=2,mode="bilinear",align_corners=False)
        #mono_disp = [mono_disp[i]+F.interpolate(2*mono_disp[i+1],scale_factor=2,mode="bilinear",align_corners=False) for i in range(len(mono_disp)-2, -1, -1)]
        #mono_disp = [last_disp] + mono_disp # small to big 

        half_feat = [left_feat2, right_feat2]
        stereo_disps = self.disp_head(half_feat, mono_disp[0][:B,...]) # input mono disp /4 resolution
        if isinstance(stereo_disps, (list,tuple)):
            stereo_disps = stereo_disps[0]
        return  (mono_disp, stereo_disps)


    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        B = left_imgs.shape[0]
        mono_disp, stereo_disp = self.extract_disp(left_imgs, right_imgs,  **kwargs)

        # mono loss
        #gt_depth = (kwargs["focal"]*kwargs["baseline"]).reshape(B,1,1,1) / (kwargs["left_disps"] + 1e-6)
        gt_stereo_disp = torch.cat([kwargs["left_disps"], kwargs["right_disps"]], 0)
        #(kwargs['focal']*kwargs['baseline']).reshape(B,1,1,1).repeat(2,1,1,1)  / (1e-6+gt_stereo_disp)
        loss_mono = self.mono_head.loss(mono_disp, gt_stereo_disp)
        losses.update(loss_mono)
        # stereo loss
        loss_stereo = self.disp_head.loss(stereo_disp, kwargs["left_disps"])
        losses.update(loss_stereo)
 
        return losses