import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings
import torch.distributed as dist
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from hdvo.core.visulization import vis_depth_tensor,vis_img_tensor
from ..registry import HYBRID_METHOD
from ..builder import build,build_loss,build_head,build_backbone,build_neck,build_mono_predictor,build_stereo_predictor,build_visual_odometry,build_mask

@HYBRID_METHOD.register_module()
class HDVO(nn.Module):
    def __init__(self, depth_net=None, 
                 segment_net=None,
                 pose_net=None, 
                 brightness_net=None, 
                 occ_mask=None,
                 homo_mask=None,
                 stereo_head=None,
                 ddvo_head=None, 
                 smooth_loss=None,
                 ddvo_module=None,
                 use_stereo_depth=True,
                 use_TTT = False,
                 use_sup_pose=False,
                 bidirection = False,
                 use_stereo_prediction=False,
                 GaussianNewtonLoss=False,
                 **kwargs):
        super(HDVO, self).__init__()
        self.GaussianNewtonLoss = GaussianNewtonLoss
        self.bidirection = bidirection
        self.use_sup_pose = use_sup_pose
        self.use_TTT = use_TTT
        self.use_stereo_prediction = use_stereo_prediction
        if use_TTT:
            self.iter = 0
            self.initcTr, self.initrTc = nn.Parameter(kwargs['initcTr'], requires_grad=True), nn.Parameter(kwargs['initrTc'], requires_grad=True)
        self.use_stereo_depth = use_stereo_depth
        if depth_net is not None:  
            if use_stereo_depth: self.depth_net = build_stereo_predictor(depth_net)
            else: self.depth_net = build_mono_predictor(depth_net)
        else: self.depth_net = None
        if segment_net is not None:  self.segment_net = build_stereo_predictor(segment_net)
        else: self.segment_net = None
        if pose_net is not None:  self.pose_net = build_visual_odometry(pose_net)
        else: 
            self.pose_net = None
        self.save_last_pose = None
        if brightness_net is not None:  self.brightness_net = build_visual_odometry(brightness_net)
        else: self.brightness_net = None
        if stereo_head is not None:  self.stereo_head = build_head(stereo_head) 
        else: self.stereo_head = None
        if ddvo_head is not None:  self.ddvo_head = build_head(ddvo_head)
        else: self.ddvo_head = None
        if ddvo_module is not None:  self.ddvo_module = build_visual_odometry(ddvo_module)
        else: self.ddvo_module = None
        if occ_mask is not None:  self.occ_mask = build_mask(occ_mask)
        else: self.occ_mask = None
        if homo_mask is not None: self.homo_mask = build_mask(homo_mask)
        else: self.homo_mask = None

        if smooth_loss is not None:
            self.smooth_loss = build_loss(smooth_loss)
        else:
            self.smooth_loss = None


    def forward_train(self, left_imgs, right_imgs=None, **kwargs):
        if self.use_stereo_depth:
            if self.bidirection:
                return self.forward_train_stereo_bi(left_imgs, right_imgs, **kwargs)
            return self.forward_train_stereo(left_imgs, right_imgs, **kwargs)
        else:
            return self.forward_train_mono(left_imgs, right_imgs, **kwargs)

    def forward_test(self, left_imgs, right_imgs=None, **kwargs):
        if self.use_stereo_depth:
            return self._do_test( left_imgs, right_imgs,**kwargs)
        else:
            return self._do_test_mono(left_imgs, right_imgs, **kwargs)


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

    def forward(self, left_imgs, right_imgs=None,  return_loss=True,  **kwargs):
        #focal=None, baseline=None, intrinsics=None,
        if return_loss:
            return self.forward_train(left_imgs, right_imgs, **kwargs)
        #if focal is not None: kwargs['focal'] = focal
        #if baseline is not None: kwargs['baseline'] = baseline
        #if intrinsics is not None: kwargs['intrinsics'] = intrinsics
        return self.forward_test(left_imgs, right_imgs,  **kwargs)


    def train_step(self, data_batch, optimizer, **kwargs):
        left_imgs = data_batch['left_imgs']
        right_imgs = data_batch['right_imgs'] if "right_imgs" in data_batch else None
        
        aux_info = {}
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        if "right_imgs" in data_batch:
            keys.remove('right_imgs')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)
        if self.GaussianNewtonLoss: 
            l2_losses = dict()
            for k,v in losses.items():
                l2_losses[k] = 0.5 * losses[k] ** 2
        loss, log_vars = self._parse_losses(losses)
        if self.GaussianNewtonLoss: l2_loss, log_vars = self._parse_losses(l2_losses)
        
        if self.GaussianNewtonLoss:
            outputs = dict(
                loss=loss,
                l2_loss=l2_loss,
                log_vars=log_vars,
                num_samples=len(next(iter(data_batch.values()))))
        else:
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
        left_imgs = data_batch['left_imgs']
        right_imgs = data_batch['right_imgs'] if "right_imgs" in data_batch else None
        
        aux_info = {}
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        if "right_imgs" in data_batch:
            keys.remove('right_imgs')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    
