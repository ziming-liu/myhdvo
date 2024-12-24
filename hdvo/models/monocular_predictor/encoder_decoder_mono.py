'''
Author: Ziming Liu
Date: 2023-02-08 21:34:59
LastEditors: Ziming Liu
LastEditTime: 2023-08-13 16:28:43
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings

from ..builder import build_backbone, build_cost_processor, build_disp_predictor,build_loss

from ..registry import MONO_PREDICTOR

from ..losses import DispL1Loss
from .base_mono import BaseMono

from ...core.visulization import vis_depth_tensor,vis_img_tensor


@MONO_PREDICTOR.register_module()
class EncoderDecoderMono(BaseMono):
    def __init__(self, backbone, disp_head, neck, smooth_loss=None,
                  pretrained=None, pred_format="left_depths", pred_format_eval=None,
                    **kwargs):
        super().__init__(backbone, disp_head, neck, pretrained, **kwargs)
        self.pred_format = pred_format
        self.pred_format_eval = pred_format_eval
        if self.pred_format_eval is None:
            self.pred_format_eval = pred_format
        if smooth_loss is not None:
            self.smooth_loss_func = build_loss(smooth_loss)

    def forward_train_subnetwork(self, left_imgs, **kwargs):
        #vis_img_tensor(left_imgs[0], "/home/ziliu/vis/monodepth3", "leftimgs")
        x = self.backbone(left_imgs) # x is a list of feat pyramid
        x = self.neck(x)
        x = self.disp_head(x)
        return x

    def forward_train(self, left_imgs, **kwargs):
        #vis_img_tensor(left_imgs[0], "/home/ziliu/vis/monodepth3", "leftimgs")
        x = self.backbone(left_imgs) # x is a list of feat pyramid
        x = self.neck(x)
        x = self.disp_head(x)
        #vis_depth_tensor(x, "/home/ziliu/vis/monodepth3", "x")
        #print("pred >> ")
        ##print(x[0,0,:,100])
        #print("gt label >>")
        #print(kwargs[self.pred_format][0,0,:,100])
        losses = self.disp_head.loss(x, kwargs[self.pred_format])
        return losses

    def forward_test(self, left_imgs, **kwargs):
        x = self.backbone(left_imgs)
        x = self.neck(x)
        x = self.disp_head(x)
        outputs = [[],[],[],[],[],[]]
        if self.pred_format != self.pred_format_eval:
            x[x==0] = 1e-3
            x = kwargs["focal"] * kwargs["baseline"] / x
        outputs[0] = x.float().cpu().numpy()
        outputs[1] = kwargs[self.pred_format_eval].float().cpu().numpy()
        return outputs


    def smooth_loss(self, disp, img):
        loss = []
        if not isinstance(disp, (list, tuple)):
            disp = [disp] 
        for i in range(len(disp)):
            losses = {}
            losses[f"smooth_loss_{i}"] = self.smooth_loss_func(disp[i], img)
            loss.append(losses)
        return loss
