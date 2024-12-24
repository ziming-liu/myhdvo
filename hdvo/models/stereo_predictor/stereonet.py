

"""
Classes and functions to instantiate, and train, a StereoNet model (https://arxiv.org/abs/1807.08865).

StereoNet model is decomposed into a feature extractor, cost volume creation, and a cascade of refiner networks.

Loss function is the Robust Loss function (https://arxiv.org/abs/1701.03077)
"""

from typing import Tuple, List, Optional, Dict, Any, Callable
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

import time
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
class StereoNet(pl.LightningModule):
    """
    StereoNet model.  During training, takes in a torch.Tensor dimensions [batch, left/right/disp_left/disp_right channels, height, width].
    At inference, (ie. calling the forward method), only the predicted left disparity is returned.

    Trained with RMSProp + Exponentially decaying learning rate scheduler.
    """

    def __init__(self, in_channels: int,
                 k_downsampling_layers: int = 3,
                 k_refinement_layers: int = 3,
                 candidate_disparities: int = 256,
                 feature_extractor_filters: int = 32,
                 cost_volumizer_filters: int = 32,
                 mask: bool = True,
                 optimizer_partial: Optional[Callable[[torch.nn.Module], torch.optim.Optimizer]] = None,
                 scheduler_partial: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.k_downsampling_layers = k_downsampling_layers
        self.k_refinement_layers = k_refinement_layers
        self.candidate_disparities = candidate_disparities
        self.mask = mask

        self.feature_extractor_filters = feature_extractor_filters
        self.cost_volumizer_filters = cost_volumizer_filters

        self._max_downsampled_disps = (candidate_disparities+1) // (2**k_downsampling_layers)

        # Feature network
        self.feature_extractor = FeatureExtractor(in_channels=in_channels, out_channels=self.feature_extractor_filters, k_downsampling_layers=self.k_downsampling_layers)

        # Cost volume
        self.cost_volumizer = CostVolume(in_channels=self.feature_extractor_filters, out_channels=self.cost_volumizer_filters, max_downsampled_disps=self._max_downsampled_disps)

        # Hierarchical Refinement: Edge-Aware Upsampling
        self.refiners = nn.ModuleList()
        for _ in range(self.k_refinement_layers):
            self.refiners.append(Refinement(in_channels=in_channels+1))

        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)


    def forward_pyramid(self, sample: torch.Tensor, side: str = 'left') -> List[torch.Tensor]:
        """
        This is the heart of the forward pass.  Given a torch.Tensor of shape [Batch, left/right, Height, Width], perform the feature extraction, cost volume estimation, cascading
        refiners to return a list of the disparities.  First entry of the returned list is the lowest resolution while the last is the full resolution disparity.

        For clarity, the zeroth element of the first dimension is the left image and the first element of the first dimension is the right image.

        The idea with reference/shifting is that when computing the cost volume, one image is effectively held stationary while the other image
        sweeps across.  If the provided tuple (x) is (left/right) stereo pair with the argument side='left', then the stationary image will be the left
        image and the sweeping image will be the right image and vice versa.
        """
        if side == 'left':
            reference = sample[:, :self.in_channels, ...]
            shifting = sample[:, self.in_channels:self.in_channels*2, ...]
        elif side == 'right':
            reference = sample[:, self.in_channels:self.in_channels*2, ...]
            shifting = sample[:, :self.in_channels, ...]

        reference_embedding = self.feature_extractor(reference)
        shifting_embedding = self.feature_extractor(shifting)

        cost = self.cost_volumizer((reference_embedding, shifting_embedding), side=side)

        disparity_pyramid = [soft_argmin(cost, self.candidate_disparities)]

        for idx, refiner in enumerate(self.refiners, start=1):
            scale = (2**self.k_refinement_layers) / (2**idx)
            new_h, new_w = int(reference.size()[2]//scale), int(reference.size()[3]//scale)
            reference_rescaled = F.interpolate(reference, [new_h, new_w], mode='bilinear', align_corners=True)
            disparity_low_rescaled = F.interpolate(disparity_pyramid[-1], [new_h, new_w], mode='bilinear', align_corners=True)
            refined_disparity = F.relu(refiner(torch.cat((reference_rescaled, disparity_low_rescaled), dim=1)) + disparity_low_rescaled)
            disparity_pyramid.append(refined_disparity)

        return disparity_pyramid

    def forward_test(self, left_imgs, right_imgs, **kwargs):
        """
        Do the forward pass using forward_pyramid (for the left disparity map) and return only the full resolution map.
        """
        t0 = time.time()
        gt_disparity_left = kwargs['left_disps']
        #gt_disparity_right = kwargs['right_disps']
        batch = torch.cat((left_imgs, right_imgs,gt_disparity_left), dim=1)
        outputs = [[], [], [],[],[],[]]
        disparities = self.forward_pyramid(batch, side='left')
        outputs[0] = disparities[-1].detach().cpu().numpy()
        outputs[1] = gt_disparity_left.detach().cpu().numpy()
        torch.cuda.synchronize()
        t2 = time.time()
        if self.timer["f0_time"] == 0:
            self.timer["f0_time"] = t2 - t0
        else:
            self.timer["sum_time"] += t2 - t0
            self.timer["count"] += 1
            self.timer["avg_time"] = self.timer["sum_time"] / self.timer["count"]
            self.timer["fps"] = 1 / self.timer["avg_time"]
            print("avg_time: ", self.timer["avg_time"])
            print("fps: ", self.timer["fps"])
        
        return outputs

    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """
        Compute the disparities for both the left and right volumes then compute the loss for each.  Finally take the mean between the two losses and
        return that as the final loss.

        Log at each step the Robust Loss and log the L1 loss (End-point-error) at each epoch.
        """
        gt_disparity_left = kwargs['left_disps']
        gt_disparity_right = kwargs['right_disps']
        batch = torch.cat((left_imgs, right_imgs,gt_disparity_left,gt_disparity_right), dim=1)

        height, width = batch.size()[-2:]

        # Non-uniform because the sizes of each of the list entries returned from the forward_pyramid aren't the same
        disp_pred_left_nonuniform = self.forward_pyramid(batch, side='left')
        disp_pred_right_nonuniform = self.forward_pyramid(batch, side='right')

        for idx, (disparity_left, disparity_right) in enumerate(zip(disp_pred_left_nonuniform, disp_pred_right_nonuniform)):
            disp_pred_left_nonuniform[idx] = F.interpolate(disparity_left, [height, width], mode='bilinear', align_corners=True)
            disp_pred_right_nonuniform[idx] = F.interpolate(disparity_right, [height, width], mode='bilinear', align_corners=True)

        disp_pred_left = torch.stack(disp_pred_left_nonuniform, dim=0)
        disp_pred_right = torch.stack(disp_pred_right_nonuniform, dim=0)

        def _tiler(tensor: torch.Tensor, matching_size: Optional[List[int]] = None) -> torch.Tensor:
            if matching_size is None:
                matching_size = [disp_pred_left.size()[0], 1, 1, 1, 1]
            return tensor.tile(matching_size)

        disp_gt_left = _tiler(batch[:, -2, ...].squeeze()).permute(0,2,1,3,4)
        disp_gt_right = _tiler(batch[:, -1, ...].squeeze()).permute(0,2,1,3,4)
        
        if self.mask:
            left_mask = (disp_gt_left < self.candidate_disparities).detach()
            right_mask = (disp_gt_right < self.candidate_disparities).detach()

            loss_left = torch.mean(robust_loss(disp_gt_left[left_mask] - disp_pred_left[left_mask], alpha=1, c=2))
            loss_right = torch.mean(robust_loss(disp_gt_right[right_mask] - disp_pred_right[right_mask], alpha=1, c=2))
        else:
            loss_left = torch.mean(robust_loss(disp_gt_left - disp_pred_left, alpha=1, c=2))
            loss_right = torch.mean(robust_loss(disp_gt_right - disp_pred_right, alpha=1, c=2))

        loss = dict()
        loss['left'] = loss_left/2
        loss['right'] = loss_right/2
        #loss = (loss_left + loss_right) / 2
        #loss = loss_left

        #self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        #self.log("train_loss_epoch", F.l1_loss(disp_pred_left[-1], disp_gt_left[-1]), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def forward(self, left_imgs, right_imgs =None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        #assert "left_disps" in kwargs and "right_disps" in kwargs
        if return_loss:
            return self.forward_train(left_imgs, right_imgs, **kwargs)

        return self.forward_test(left_imgs, right_imgs, **kwargs)


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
        #self.start_index = kwargs['start_index']
        #self.seq_length =  kwargs['seq_len']
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(left_imgs, right_imgs, **kwargs)
        if return_loss:
            #if pose is None:
            #    return self.forward_train(left_imgs, right_imgs, **kwargs)
            #    #raise ValueError('Label should not be None.')
            return self.forward_train(left_imgs, right_imgs, **kwargs)

        return self.forward_test(left_imgs, right_imgs, **kwargs)

    def extract_feature(self, left_img, right_img):
        if self.shared_backbone:
            return (self.backbone(left_img), self.backbone(right_img))
        else:
            return (self.bacbkone1(left_img),self.backbone2(right_img))
 
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
        #    #print('-->name:', name, '-->grad_requirs:',parms.requires_grad,  )
        #    #print(' -->grad_value: \n {}'.format(parms.grad))

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

        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
        # print gradient of network
        #for name, parms in self.named_parameters():
        #    #print('-->name:', name, '-->grad_requirs:',parms.requires_grad,  )
        #    #print(' -->grad_value: \n {}'.format(parms.grad))

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    


class FeatureExtractor(torch.nn.Module):
    """
    Feature extractor network with 'K' downsampling layers.  Refer to the original paper for full discussion.
    """

    def __init__(self, in_channels: int, out_channels: int, k_downsampling_layers: int):
        super().__init__()
        self.k = k_downsampling_layers

        net: OrderedDict[str, nn.Module] = OrderedDict()

        for block_idx in range(self.k):
            net[f'segment_0_conv_{block_idx}'] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
            in_channels = out_channels

        for block_idx in range(6):
            net[f'segment_1_res_{block_idx}'] = ResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        net['segment_2_conv_0'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name, missing-function-docstring
        x = self.net(x)
        return x


class CostVolume(torch.nn.Module):
    """
    Computes the cost volume and filters it using the 3D convolutional network.  Refer to original paper for a full discussion.
    """

    def __init__(self, in_channels: int, out_channels: int, max_downsampled_disps: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._max_downsampled_disps = max_downsampled_disps

        net: OrderedDict[str, nn.Module] = OrderedDict()

        for block_idx in range(4):
            net[f'segment_0_conv_{block_idx}'] = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            net[f'segment_0_bn_{block_idx}'] = nn.BatchNorm3d(num_features=out_channels)
            net[f'segment_0_act_{block_idx}'] = nn.LeakyReLU(negative_slope=0.2)  # Not clear in paper if default or implied to be 0.2 like the rest

            in_channels = out_channels

        net['segment_1_conv_0'] = nn.Conv3d(in_channels=out_channels, out_channels=1, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], side: str = 'left') -> torch.Tensor:  # pylint: disable=invalid-name
        """
        The cost volume effectively holds one of the left/right images constant (albeit clipping) and computes the difference with a
        shifting (left/right) portion of the corresponding image.  By default, this method holds the left image stationary and sweeps the right image.

        To compute the cost volume for holding the right image stationary and sweeping the left image, use side='right'.
        """
        reference_embedding, target_embedding = x

        cost = compute_volume(reference_embedding, target_embedding, max_downsampled_disps=self._max_downsampled_disps, side=side)

        cost = self.net(cost)
        cost = torch.squeeze(cost, dim=1)

        return cost


def compute_volume(reference_embedding: torch.Tensor, target_embedding: torch.Tensor, max_downsampled_disps: int, side: str = 'left') -> torch.Tensor:
    """
    Refer to the doc string in CostVolume.forward.
    Refer to https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py

    This difference based cost volume is also reflected in an implementation of the popular DispNetCorr:
        Line 81 https://github.com/wyf2017/DSMnet/blob/b61652dfb3ee84b996f0ad4055eaf527dc6b965f/models/util_conv.py
    """
    batch, channel, height, width = reference_embedding.size()
    cost = torch.Tensor(batch, channel, max_downsampled_disps, height, width).zero_()
    cost = cost.type_as(reference_embedding)  # PyTorch Lightning handles the devices
    cost[:, :, 0, :, :] = reference_embedding - target_embedding
    for idx in range(1, max_downsampled_disps):
        if side == 'left':
            cost[:, :, idx, :, idx:] = reference_embedding[:, :, :, idx:] - target_embedding[:, :, :, :-idx]
        if side == 'right':
            cost[:, :, idx, :, :-idx] = reference_embedding[:, :, :, :-idx] - target_embedding[:, :, :, idx:]
    cost = cost.contiguous()

    return cost


class Refinement(torch.nn.Module):
    """
    Several of these classes will be instantiated to perform the *cascading* refinement.  Refer to the original paper for a full discussion.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        dilations = [1, 2, 4, 8, 1, 1]

        net: OrderedDict[str, nn.Module] = OrderedDict()

        net['segment_0_conv_0'] = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)

        for block_idx, dilation in enumerate(dilations):
            net[f'segment_1_res_{block_idx}'] = ResBlock(in_channels=32, out_channels=32, kernel_size=3, padding=dilation, dilation=dilation)

        net['segment_2_conv_0'] = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name, missing-function-docstring
        x = self.net(x)
        return x


class ResBlock(torch.nn.Module):
    """
    Just a note, in the original paper, there is no discussion about padding; however, both the ZhiXuanLi and the X-StereoLab implementation using padding.
    This does make sense to maintain the image size after the feature extraction has occured.

    X-StereoLab uses a simple Res unit with a single conv and summation while ZhiXuanLi uses the original residual unit implementation.
    This class also uses the original implementation with 2 layers of convolutions.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """
        Original Residual Unit: https://arxiv.org/pdf/1603.05027.pdf (Fig 1. Left)
        """

        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activation_1(res)
        res = self.conv_2(res)
        res = self.batch_norm_2(res)

        # I'm not really sure why the type definition is required here... nn.Conv2d already returns type Tensor...
        # So res should be of type torch.Tensor AND x is already defined as type torch.Tensor.
        out: torch.Tensor = res + x
        out = self.activation_2(out)

        return out


def soft_argmin(cost: torch.Tensor, max_downsampled_disps: int) -> torch.Tensor:
    """
    Soft argmin function described in the original paper.  The disparity grid creates the first 'd' value in equation 2 while
    cost is the C_i(d) term.  The exp/sum(exp) == softmax function.
    """
    disparity_softmax = F.softmax(-cost, dim=1)
    # TODO: Bilinear interpolate the disparity dimension back to D to perform the proper d*exp(-C_i(d))

    disparity_grid = torch.linspace(0, max_downsampled_disps, disparity_softmax.size(1)).reshape(1, -1, 1, 1)
    disparity_grid = disparity_grid.type_as(disparity_softmax)

    disp = torch.sum(disparity_softmax * disparity_grid, dim=1, keepdim=True)

    return disp


def robust_loss(x: torch.Tensor, alpha: float, c: float) -> torch.Tensor:  # pylint: disable=invalid-name
    """
    A General and Adaptive Robust Loss Function (https://arxiv.org/abs/1701.03077)
    """
    f: torch.Tensor = (abs(alpha - 2) / alpha) * (torch.pow(torch.pow(x / c, 2)/abs(alpha - 2) + 1, alpha/2) - 1)  # pylint: disable=invalid-name
    return f