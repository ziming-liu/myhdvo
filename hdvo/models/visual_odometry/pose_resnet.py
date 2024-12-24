import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp
from mmcv.runner import auto_fp16
import warnings
import torch.distributed as dist
from collections import OrderedDict

from ...utils import get_root_logger
from ..builder import build_backbone, build_neck, build_disp_predictor,build_loss,build_head

from ..registry import VISUAL_ODOMETRY
from .pose_transform import *

@VISUAL_ODOMETRY.register_module
class PoseResNet(nn.Module):
    def __init__(self, backbone, head, 
                 ddvo_mode = True,
                 weights = 1.0,
                 pretrained = None,
                 subnetwork=False):
        super(PoseResNet, self).__init__()
        self.ddvo_mode = ddvo_mode # for ddvo, unsupervised training mode
        self.pretrained = pretrained
        self.subnetwork = subnetwork
        self.weights = weights

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

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
    
    def forward_train(self, left_imgs, right_imgs=None, **kwargs):
        assert len(left_imgs.shape) == 5
        batch_size, seq_len, c, h, w = left_imgs.shape
        itorchut = left_imgs.reshape(batch_size, seq_len*c, h, w)
        features = self.backbone(itorchut)
        angle, translation = self.head(features)
        angle, translation = angle.squeeze(2).squeeze(1), translation.squeeze(2).squeeze(1)

        loss = dict()
        assert 'pose' in kwargs.keys(), "pose_net is None, please provide gt pose"
        abs_poses = kwargs['pose'] # B T 4 4 load gt pose
        #initcTr = torch.linalg.solve(abs_poses[:,1,:,:], abs_poses[:,0,:,:])
        gt = (torch.linalg.inv(abs_poses[:,1,:,:].double()) @ abs_poses[:,0,:,:].double()).float()
        gt_angle = torch.stack([ torch_mat2euler(gt[i, :3, :3]) for i in range(gt.shape[0])  ])
        gt_translation = gt[:, :3, 3]

        loss["l2_pose_loss"] = self.loss(torch.cat([angle, translation],dim=1), \
                                         torch.cat([gt_angle, gt_translation],dim=1) )
        return loss

    def forward_train_subnetwork(self, left_imgs, right_imgs=None, **kwargs):
        assert len(left_imgs.shape) == 5
        batch_size, seq_len, c, h, w = left_imgs.shape
        itorchut = left_imgs.reshape(batch_size, seq_len*c, h, w)
        features = self.backbone(itorchut)
        angle, translation = self.head(features)
        angle, translation = angle.squeeze(2).squeeze(1), translation.squeeze(2).squeeze(1)
        
        #q = euler_to_quaternion(angle[:, 0:1], angle[:, 1:2], angle[:, 2:3])
        #Rt = transformation_matrix(translation, q)
        R = torch.stack([torch_euler2mat(angle[i], isRadian=False, seq="xyz", cuda=True) for i in range(angle.shape[0]) ] )
        I = torch.eye(4, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
        I[:,:3,:3] = R
        I[:,:3,3] = translation
        #print("\n",I)
        pose_6d = torch.cat([angle, translation], dim=1)
        return I, pose_6d
    
    def forward_test(self, left_imgs, right_imgs=None, **kwargs):
        assert len(left_imgs.shape) == 5
        batch_size, seq_len, c, h, w = left_imgs.shape
        itorchut = left_imgs.reshape(batch_size, seq_len*c, h, w)
        features = self.backbone(itorchut)
        angle, translation = self.head(features)
        return angle, translation

    def forward(self, left_imgs, right_imgs=None, return_loss=True, **kwargs):
        if self.subnetwork:
            return self.forward_train_subnetwork(left_imgs, right_imgs,  **kwargs)
        if self.ddvo_mode:
            return self.forward_test(left_imgs, right_imgs, **kwargs)
        if return_loss:
            return self.forward_train(left_imgs, right_imgs, **kwargs)
        return self.forward_test(left_imgs, right_imgs, **kwargs)

    def loss(self, pred, gt, **kwargs):
        if pred.shape[-2:] == (4,4):
            pred_angle = torch.stack([ torch_mat2euler(pred[i, :3, :3],seq='xyz') for i in range(pred.shape[0])  ])
            pred_angle = pred_angle * 180 / torch.FloatTensor([np.pi]).cuda() 
            pred_translation = pred[:, :3, 3]
        elif pred.shape[-1] == 6:
            pred_angle = pred[:, :3] #* torch.FloatTensor([np.pi]).cuda() / 180 # angle to radian 
            pred_translation = pred[:, 3:]
        if gt.shape[-2:] == (4,4):
            gt_angle = torch.stack([ torch_mat2euler(gt[i, :3, :3], seq='xyz') for i in range(gt.shape[0])  ])
            gt_translation = gt[:, :3, 3]
            gt_angle = gt_angle * 180 / torch.FloatTensor([np.pi]).cuda()
            #print("gt\n", gt)
            #print("gt_angle", gt_angle)
            #back_angle = torch.stack([ torch_euler2mat((gt_angle[i]), seq='xyz') for i in range(gt.shape[0])  ])
            #print("back\n", back_angle)
        elif gt.shape[-1] == 6:
            gt_angle = gt[:, :3] #* torch.FloatTensor([np.pi]).cuda() / 180 # angle to radian 
            gt_translation = gt[:, 3:]

        loss = dict()
        # Weighted MSE Loss
        #print("pred_angle", pred_angle)
        #print("gt_angle", gt_angle)
        #print(gt_translation)
        angle_loss = torch.nn.functional.mse_loss(pred_angle, gt_angle)
        translation_loss = torch.nn.functional.mse_loss(pred_translation, gt_translation)
        loss["l2_pose_rotation"] = (  angle_loss ) * self.weights
        loss["l2_pose_translation"] = (translation_loss) * self.weights

        return loss

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


    def train_step(self, data_batch, optimizer, **kwargs):
        left_imgs = data_batch['left_imgs']
        if "right_imgs" in data_batch: right_imgs = data_batch['right_imgs']
        
        aux_info = {}
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        if "right_imgs" in data_batch: keys.remove('right_imgs')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
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
        if "right_imgs" in data_batch: right_imgs = data_batch['right_imgs']
        
        aux_info = {}
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        if "right_imgs" in data_batch: keys.remove('right_imgs')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    
