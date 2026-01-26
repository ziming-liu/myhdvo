from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time 

from .coex_submodules.feature import Feature, FeatUp
from .coex_submodules.utils import AttentionCostVolume
from .coex_submodules.aggregation import Aggregation
from .coex_submodules.regression import Regression
from .coex_submodules.util_conv import BasicConv, Conv2x

from ..registry import STEREO_PREDICTOR
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


@STEREO_PREDICTOR.register_module()
class CoEx(nn.Module):
    def __init__(self, cfg, losses, subnetwork=False, pretrain=None, pretrain_official=None, **kwargs):
        super(CoEx, self).__init__()
        self.cfg = cfg
        self.type = self.cfg['backbone']['type']
        chans = self.cfg['backbone']['channels'][self.type]
        self.subnetwork = subnetwork

        self.D = int(self.cfg['max_disparity']/4)

        # set up the feature extraction first
        self.feature = Feature(self.cfg)
        self.up = FeatUp(self.cfg)

        self.corr_volume = cfg['corr_volume']
        if self.corr_volume:
            self.cost_volume = AttentionCostVolume(
                cfg['max_disparity'],
                chans[1]*2+self.cfg['spixel']['branch_channels'][1],
                chans[1]*2,
                1,
                weighted=cfg['matching_weighted'])
            matching_head = cfg['matching_head']
        else:
            self.cost_conv = BasicConv(
                chans[1]*2+self.cfg['spixel']['branch_channels'][1],
                chans[1]*2,
                kernel_size=3,
                padding=1,
                stride=1)
            self.cost_desc = nn.Conv2d(
                chans[1]*2,
                chans[1],
                kernel_size=1,
                padding=0,
                stride=1)
            matching_head = chans[1]*2

        self.cost_agg = Aggregation(
            cfg['backbone'],
            max_disparity=cfg['max_disparity'],
            matching_head=matching_head,
            gce=cfg['gce'],
            disp_strides=cfg['aggregation']['disp_strides'],
            channels=cfg['aggregation']['channels'],
            blocks_num=cfg['aggregation']['blocks_num'],
            spixel_branch_channels=cfg['spixel']['branch_channels'])
        self.regression = Regression(
            max_disparity=cfg['max_disparity'],
            top_k=cfg['regression']['top_k'])

        self.stem_2 = nn.Sequential(
            BasicConv(3, self.cfg['spixel']['branch_channels'][0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.cfg['spixel']['branch_channels'][0], self.cfg['spixel']['branch_channels'][0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.cfg['spixel']['branch_channels'][0]), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(self.cfg['spixel']['branch_channels'][0], self.cfg['spixel']['branch_channels'][1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.cfg['spixel']['branch_channels'][1], self.cfg['spixel']['branch_channels'][1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.cfg['spixel']['branch_channels'][1]), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(chans[1], 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(chans[1]*2+self.cfg['spixel']['branch_channels'][1], chans[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(chans[1], chans[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chans[1]), nn.ReLU()
            )
        
        self.loss_disp = builder.build_loss(losses)
        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)

        if pretrain is not None:
            self.__init_weights(pretrain)
        if pretrain_official is not None:
            self.__init_backbone_weights(pretrain_official)

    def __init_weights(self, pretrain):
        load_checkpoint(self, pretrain, map_location='cpu')
        print("load pretrain")

    def __init_backbone_weights(self, backbone_pretrain):
        checkpoint = _load_checkpoint(backbone_pretrain, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove "stereo." prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('stereo.'):
                new_key = key[7:]  # Remove "stereo." (7 characters)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the modified state_dict
        self.load_state_dict(new_state_dict, strict=False)
        print("load backbone pretrain (removed 'stereo.' prefix from keys)")


    def extract_disp(self, imL, imR=None, u0=None, v0=None, training=False):
        if imR is not None:
            assert imL.shape == imR.shape
            imL = torch.cat([imL, imR], 0)
            
        b, c, h, w = imL.shape
        v2, v = self.feature(imL)
        x2, y2 = v2.split(dim=0, split_size=b//2)

        v = self.up(v)
        x, y = [], []
        for v_ in v:
            x_, y_ = v_.split(dim=0, split_size=b//2)
            x.append(x_)
            y.append(y_)

        stem_2v = self.stem_2(imL)
        stem_4v = self.stem_4(stem_2v)
        stem_2x, stem_2y = stem_2v.split(dim=0, split_size=b//2)
        stem_4x, stem_4y = stem_4v.split(dim=0, split_size=b//2)

        x[0] = torch.cat((x[0], stem_4x), 1)
        y[0] = torch.cat((y[0], stem_4y), 1)

        # Cost volume processing

        if self.corr_volume:
            cost = (self.cost_volume(x[0], y[0]))[:, :, :-1]
        else:
            refimg_fea = self.cost_conv(x[0])
            targetimg_fea = self.cost_conv(y[0])
            refimg_fea = self.cost_desc(refimg_fea)
            targetimg_fea = self.cost_desc(targetimg_fea)

            cost = Variable(
                torch.FloatTensor(
                    refimg_fea.size()[0],
                    refimg_fea.size()[1]*2,
                    self.D, 
                    refimg_fea.size()[2], 
                    refimg_fea.size()[3]).zero_()).cuda()
            for i in range(self.D):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                    cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
            cost = cost.contiguous()

        cost = self.cost_agg(x, cost)

        # spixel guide comp
        xspx = self.spx_4(x[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        # Regression
        disp_pred = self.regression(cost, spx_pred, training=training)
        #if training:
        #    disp_pred.append(0)

        return disp_pred
    def forward_subnetwork(self, left_imgs, right_imgs, **kwargs):
        # compute outputs
        disp_preds = self.extract_disp(left_imgs, right_imgs, training=True)
        disp_preds = [a.unsqueeze(1) for a in disp_preds]
        #disp_preds = [F.interpolate(a, size=disp_preds[0].shape[-2:], mode="bilinear") for a in disp_preds]

        return disp_preds[0]
    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Forward function for training.

        Args:
            left_imgs (list[Tensor]): Multi-scale left images.
            right_imgs (list[Tensor]): Multi-scale right images.

        Returns:
            dict: Contain the loss items.
        """
        losses = dict()

        # compute outputs
        disp_preds = self.extract_disp(left_imgs, right_imgs, training=True)

        # generate ground truth
        disp_gt = kwargs['left_disps']
        disp_preds = [a.unsqueeze(1) for a in disp_preds]
        # compute losses
        loss = self.loss_disp(disp_preds, disp_gt, )
        losses.update(loss)

        return losses

    def forward_test(self, left_img, right_img, **kwargs):
        """Forward function for testing.

        Args:
            left_img (Tensor): Left image.
            right_img (Tensor): Right image.

        Returns:
            Tensor: Disparity prediction.
        """
        t0 = time.time()
        print(left_img.shape, right_img.shape)
        outputs = [[],[],[],[],[],[],[]]
        #with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        disp_preds = self.extract_disp(left_img, right_img)
        #print(prof)
        
        outputs[0] = disp_preds[0].detach().cpu().numpy()
        outputs[1] = kwargs['left_disps'].detach().cpu().numpy()
        torch.cuda.synchronize()
        t1 = time.time()
        self.timer["sum_time"] += t1 - t0
        self.timer["count"] += 1
        if self.timer["f0_time"] == 0:
            self.timer["f0_time"] = t1 - t0
        else:
            self.timer["avg_time"] = (self.timer["sum_time"]-self.timer["f0_time"]) / (self.timer["count"]-1)
            self.timer["fps"] = 1 / self.timer["avg_time"]
            print("avg_time: ", self.timer["avg_time"])
            print("fps: ", self.timer["fps"])
 
        
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
        #self.start_index = kwargs['start_index']
        #self.seq_length =  kwargs['seq_len']
        if self.subnetwork:
            return self.forward_subnetwork(left_imgs, right_imgs, **kwargs)
        
        if return_loss:
            #if pose is None:
            #    return self.forward_train(left_imgs, right_imgs, **kwargs)
            #    #raise ValueError('Label should not be None.')
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
    
