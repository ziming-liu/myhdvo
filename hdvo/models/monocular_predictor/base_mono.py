'''
Author: Ziming Liu
Date: 2022-07-08 00:04:48
LastEditors: Ziming Liu
LastEditTime: 2023-07-10 04:28:18
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from .. import builder
import warnings

from ..builder import build_backbone, build_head, build_neck, build_loss

from ..registry import MONO_PREDICTOR

from ..losses import DispL1Loss

@MONO_PREDICTOR.register_module()
class BaseMono(nn.Module):
    """
    Base depth method. 

    """
    def __init__(self, backbone, disp_head, neck=None, pretrained=None,
                 subnetwork=False, **kwargs):
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
        super(BaseMono, self).__init__()
        self.subnetwork = subnetwork
        self.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.disp_head = build_head(disp_head)
        if neck is not None:
            self.neck = build_neck(neck)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @abstractmethod
    def forward_train(self, left_imgs, **kwargs):
        """Defines the computation performed at every call when training."""

    @abstractmethod
    def forward_test(self, left_imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""

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

    def forward(self, left_imgs, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        #self.start_index = kwargs['start_index']
        #self.seq_length =  kwargs['seq_len']
        #if kwargs.get('gradcam', False):
        #    del kwargs['gradcam']
        #    return self.forward_gradcam(left_imgs, **kwargs)
        if return_loss:
            if self.subnetwork:
                return self.forward_train_subnetwork(left_imgs, **kwargs)
            return self.forward_train(left_imgs, **kwargs)

        return self.forward_test(left_imgs, **kwargs)

    def extract_feature(self, left_img):
        return self.backbone(left_img)


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
        left_imgs = data_batch['left_imgs']#, data_batch['right_imgs']
        #label = data_batch['pose']
        
        aux_info = {}
        # if aux info is not define in config, there 
        #for item in self.aux_info:
        #    assert item in data_batch
        #    aux_info[item] = data_batch[item]
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        #keys.remove('right_imgs')
        #keys.remove('pose')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, return_loss=True, **aux_info)

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
        left_imgs = data_batch['left_imgs']
        #label = data_batch['pose']

        aux_info = {}
        #for item in self.aux_info:
        #    aux_info[item] = data_batch[item]
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        #keys.remove('right_imgs')
        #keys.remove('pose')
        for item in keys:
            aux_info[item] = data_batch[item]

            
        outputs = self(left_imgs, return_loss=False, **aux_info)

        return outputs

    
