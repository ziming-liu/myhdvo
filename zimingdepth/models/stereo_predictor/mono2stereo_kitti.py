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
 
import torch.distributed as dist
from mmcv.runner import auto_fp16
from .. import builder
from ..builder import build_head,build_backbone,build_neck

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor

@STEREO_PREDICTOR.register_module()
class Mono2StereoKITTI(nn.Module):
    """
    Base depth method. 

    """
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head,  pretrained=None, output_branch="stereo", prediction_format="left_disps", **kwargs):
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
        super(Mono2StereoKITTI, self).__init__()

        self.data = [] # num of frames, each component is a dict{}
        self.pretrained =pretrained
        self._MAX_DISP= -1
        self.prediction_format = prediction_format
        self.output_branch = output_branch
        self.mono_head = build_head(mono_head)      
        self.disp_head = build_head(disp_head)  
        self.mono_neck = build_neck(mono_neck)
        self.mono_backbone = build_backbone(mono_backbone)
        #self.stereo_backbone_left = build_backbone(stereo_backbone)
        self.stereo_backbone_right = build_backbone(stereo_backbone)
        self.init_weights()
    
    def init_weights(self,):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self, self.pretrained, strict=False, logger=logger)
            print(f"loaded pretrained depth_model {self.pretrained}!!!")



    def extract_disp(self, left_img, right_img, **kwargs):
        B,_,H,W = left_img.shape
        left_feat = self.mono_backbone(left_img)
        #left_feat2 = self.stereo_backbone_left(left_img)
        left_feat2 = [left_feat[1]]
        right_feat = self.stereo_backbone_right(right_img)
        if not isinstance(right_feat, (list,tuple)):
            right_feat = [right_feat]
        assert len(left_feat)>1, "backbone gives 4 feats"
        # mono branch 
        mono_feat = self.mono_neck(list(left_feat[1:]))
        if len(mono_feat)>1: # FPN outputs ()
            mono_feat = mono_feat[0]
        #mono_feat = F.interpolate(mono_feat, size=(H,W), mode="bilinear", align_corners=False)
        mono_depth = self.mono_head(mono_feat)
        mono_depth = F.interpolate(mono_depth, size=(H,W),mode="bilinear", align_corners=False)
        mono_disp = (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/(mono_depth+1e-6)
        #print(left_feat2[1].shape)
        #exit()
        half_feat = [left_feat2, right_feat]
        stereo_disps = self.disp_head(half_feat, mono_disp)
        if isinstance(stereo_disps, (list,tuple)):
            stereo_disps = stereo_disps[0]
        return  (mono_depth, stereo_disps)


    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        B = left_imgs.shape[0]
        mono_depth, stereo_disp = self.extract_disp(left_imgs, right_imgs, **kwargs)

        # mono loss
        loss_mono = self.mono_head.loss(mono_depth, kwargs["left_depths"])
        losses.update(loss_mono)
        # stereo loss
        loss_stereo = self.disp_head.loss(stereo_disp, (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/ (1e-6+kwargs["left_depths"]))
        losses.update(loss_stereo)
 
        return losses

 
    def forward_test(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        

        return self._do_test( left_imgs, right_imgs,**kwargs)
        
    def _do_test(self, left_imgs, right_imgs,**kwargs):
        """Defines the computation performed at every call when training."""
        B = left_imgs.shape[0]
        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir
        mono_depth, stereo_disp = self.extract_disp(left_imgs, right_imgs, **kwargs)
        if self.output_branch == "stereo":
            if self.prediction_format != "left_disps":
                stereo_disp = (kwargs["focal"]*kwargs["baseline"]).reshape(B,1,1,1) / (stereo_disp+1e-6)
            outputs[0] = stereo_disp.detach().float().cpu().numpy()
        else:
            outputs[0] = mono_depth.detach().float().cpu().numpy()
        if self.prediction_format in kwargs:
            outputs[1] = kwargs[self.prediction_format].detach().float().cpu().numpy()
        
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

    def forward(self, left_imgs, right_imgs, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
 
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
class Mono2StereoKITTILite(Mono2StereoKITTI):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)
        #self.stereo_backbone_right =None

    def extract_disp(self, left_img, right_img, **kwargs):
        B,_,H,W = left_img.shape
        #joint_imgs = torch.cat([left_img, right_img],0)
        left_feat = self.mono_backbone(left_img)
        #left_feat = [joint_feat[i][:B,...] for i in range(len(joint_feat))]
        #left_feat2= [joint_feat[1][:B,...]]
        #right_feat = [joint_feat[1][B:,...]]
        #left_feat2 = self.stereo_backbone_left(left_img)
        left_feat2 = [left_feat[1]]
        right_feat = self.stereo_backbone_right(right_img)
        if not isinstance(right_feat, (list,tuple)):
            right_feat = [right_feat]
        assert len(left_feat)>1, "backbone gives 4 feats"
        # mono branch 
        mono_feat = self.mono_neck(list(left_feat[1:]))
        #print(mono_feat.shape)
        if len(mono_feat)>1: # FPN outputs ()
            mono_feat = mono_feat[0]
        #mono_feat = F.interpolate(mono_feat, size=(H,W), mode="bilinear", align_corners=False)
        mono_depth = self.mono_head(mono_feat)
        mono_disp = (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/(mono_depth+1e-6)
        mono_depth = F.interpolate(mono_depth, size=(H,W),mode="bilinear", align_corners=False)
        #print(left_feat2[1].shape)
        #exit()
        half_feat = [left_feat2, right_feat]
        stereo_disps = self.disp_head(half_feat, mono_disp)
        if isinstance(stereo_disps, (list,tuple)):
            stereo_disps = stereo_disps[0]
        stereo_disps = 4* F.interpolate(stereo_disps, size=(H,W), mode="bilinear", align_corners=False)
        return  (mono_depth, stereo_disps)


@STEREO_PREDICTOR.register_module()
class Mono2StereoKITTILiteUnsup(Mono2StereoKITTI):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, unsup_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)
        #self.stereo_backbone_right =None
        self.unsup_head= build_head(unsup_head)

    def extract_disp(self, left_img, right_img, **kwargs):
        B,_,H,W = left_img.shape
        #joint_imgs = torch.cat([left_img, right_img],0)
        left_feat = self.mono_backbone(left_img)
        #left_feat = [joint_feat[i][:B,...] for i in range(len(joint_feat))]
        #left_feat2= [joint_feat[1][:B,...]]
        #right_feat = [joint_feat[1][B:,...]]
        #left_feat2 = self.stereo_backbone_left(left_img)
        left_feat2 = [left_feat[1]]
        right_feat = self.stereo_backbone_right(right_img)
        if not isinstance(right_feat, (list,tuple)):
            right_feat = [right_feat]
        assert len(left_feat)>1, "backbone gives 4 feats"
        # mono branch 
        mono_feat = self.mono_neck(list(left_feat[1:]))
        #print(mono_feat.shape)
        if len(mono_feat)>1: # FPN outputs ()
            mono_feat = mono_feat[0]
        #mono_feat = F.interpolate(mono_feat, size=(H,W), mode="bilinear", align_corners=False)
        mono_depth = self.mono_head(mono_feat)
        mono_disp = (0.25* kwargs["focal"] * kwargs["baseline"]).reshape((B,1,1,1))/(mono_depth+1e-6)
        mono_depth = F.interpolate(mono_depth, size=(H,W),mode="bilinear", align_corners=False)
        #print(left_feat2[1].shape)
        #exit()
        half_feat = [left_feat2, right_feat]
        stereo_disps = self.disp_head(half_feat, mono_disp)
        if isinstance(stereo_disps, (list,tuple)):
            stereo_disps = stereo_disps[0]
        stereo_disps = 4* F.interpolate(stereo_disps, size=(H,W), mode="bilinear", align_corners=False)
        return  (mono_depth, stereo_disps)

    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        B = left_imgs.shape[0]
        mono_depth, stereo_disp = self.extract_disp(left_imgs, right_imgs, **kwargs)
        
        # mono loss
        loss_mono = self.mono_head.loss(mono_depth, kwargs["left_depths"])
        losses.update(loss_mono)
        # stereo loss
        loss_stereo = self.disp_head.loss(stereo_disp, (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/ (1e-6+kwargs["left_depths"]))
        losses.update(loss_stereo)

        # unsupervised optimization 
        mono_warped_leftimg, mono_warped_rightimg =  self.unsup_head(left_imgs, right_imgs, (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/(1e-6+mono_depth))
        stereo_warped_leftimg, stereo_warped_rightimg = self.unsup_head(left_imgs, right_imgs, stereo_disp)
        loss_unsup_mono = self.unsup_head.loss(mono_warped_leftimg, left_imgs, "monoheadL")
        losses.update(loss_unsup_mono)
        loss_unsup_stereo = self.unsup_head.loss(stereo_warped_leftimg, left_imgs, "stereoheadL")
        losses.update(loss_unsup_stereo)

        return losses

@STEREO_PREDICTOR.register_module()
class InsertedMono2StereoKITTILite(Mono2StereoKITTI):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)


    def extract_disp(self, left_img, right_img, **kwargs):
        B,_,H,W = left_img.shape
        #joint_imgs = torch.cat([left_img, right_img],0)
        left_feat = self.mono_backbone(left_img)
        #left_feat = [joint_feat[i][:B,...] for i in range(len(joint_feat))]
        #left_feat2= [joint_feat[1][:B,...]]
        #right_feat = [joint_feat[1][B:,...]]
        #left_feat2 = self.stereo_backbone_left(left_img)
        left_feat2 = [left_feat[1]]
        right_feat = self.stereo_backbone_right(right_img)
        if not isinstance(right_feat, (list,tuple)):
            right_feat = [right_feat]
        assert len(left_feat)>1, "backbone gives 4 feats"
        # mono branch 
        mono_feat = self.mono_neck(list(left_feat[1:]))
        #print(mono_feat.shape)
        if len(mono_feat)>1: # FPN outputs ()
            mono_feat = mono_feat[0]
        #mono_feat = F.interpolate(mono_feat, size=(H,W), mode="bilinear", align_corners=False)
        mono_depth = self.mono_head(mono_feat)
        mono_disp = (0.25* kwargs["focal"] * kwargs["baseline"]).squeeze()/(mono_depth+1e-6)
        mono_depth = F.interpolate(mono_depth, size=(H,W),mode="bilinear", align_corners=False)
        #print(left_feat2[1].shape)
        #exit()
        half_feat = [left_feat2, right_feat]
        stereo_disps = self.disp_head(half_feat, mono_disp)
        if isinstance(stereo_disps, (list,tuple)):
            stereo_disps = stereo_disps[0]
        stereo_disps = 4* F.interpolate(stereo_disps, size=(H,W), mode="bilinear", align_corners=False)
        return  (mono_depth, stereo_disps)
        
    def forward(self, left_imgs, right_imgs, **kwargs):
        B = left_imgs.shape[0]
        results = dict()
        mono_depth, stereo_disp = self.extract_disp(left_imgs, right_imgs, **kwargs)

        results = {"mono_disps":  (kwargs["focal"]*kwargs["baseline"]).squeeze()/(1e-6+mono_depth), "disps": stereo_disp, "right_disps": None }
 
        return results

    
 ################## 
@STEREO_PREDICTOR.register_module()
class Mono2StereoKITTIUnsup(Mono2StereoKITTI):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, unsup_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", TTT=False, **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)
        self.unsup_head = build_head(unsup_head)
        self.TTT = TTT

    
    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        B = left_imgs.shape[0]
        mono_depth, stereo_disp = self.extract_disp(left_imgs, right_imgs, **kwargs)

        # mono loss
        if not self.TTT:
            loss_mono = self.mono_head.loss(mono_depth, kwargs["left_depths"])
            losses.update(loss_mono)
            # stereo loss
            loss_stereo = self.disp_head.loss(stereo_disp, (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/ (1e-6+kwargs["left_depths"]))
            losses.update(loss_stereo)

        # unsupervised optimization 
        mono_warped_leftimg, mono_warped_rightimg =  self.unsup_head(left_imgs, right_imgs, (kwargs["focal"]*kwargs["baseline"]).reshape((B,1,1,1))/(1e-6+mono_depth))
        stereo_warped_leftimg, stereo_warped_rightimg = self.unsup_head(left_imgs, right_imgs, stereo_disp)
        loss_unsup_mono = self.unsup_head.loss(mono_warped_leftimg, left_imgs, "monoheadL")
        losses.update(loss_unsup_mono)
        loss_unsup_stereo = self.unsup_head.loss(stereo_warped_leftimg, left_imgs, "stereoheadL")
        losses.update(loss_unsup_stereo)

        return losses

@STEREO_PREDICTOR.register_module()
class InsertedMono2StereoKITTI(Mono2StereoKITTI):
    def __init__(self, stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained=None, output_branch="stereo", prediction_format="left_disps", **kwargs):
        super().__init__(stereo_backbone, mono_backbone, mono_neck, disp_head, mono_head, pretrained, output_branch, prediction_format, **kwargs)
        self.pretrained = pretrained
        self.init_weights()
    
    def init_weights(self,):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self, self.pretrained, strict=False, logger=logger)
            print(f"loaded pretrained depth_model {self.pretrained}!!!")


    def forward(self, left_imgs, right_imgs, **kwargs):
        B = left_imgs.shape[0]
        results = dict()
        mono_depth, stereo_disp = self.extract_disp(left_imgs, right_imgs, **kwargs)

        results = {"mono_disps":  (kwargs["focal"]*kwargs["baseline"]).squeeze()/(1e-6+mono_depth), "disps": stereo_disp, "right_disps": None }
 
        return results
