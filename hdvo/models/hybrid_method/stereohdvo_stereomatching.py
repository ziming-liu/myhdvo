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
from .hdvo import HDVO

@HYBRID_METHOD.register_module()
class StereoHDVOStereomatching(HDVO):
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
                 **kwargs):
        super(StereoHDVOStereomatching, self).__init__(depth_net, segment_net, pose_net, brightness_net, 
                                       occ_mask, homo_mask, stereo_head, ddvo_head, smooth_loss, ddvo_module, 
                                       use_stereo_depth, use_TTT, use_sup_pose, bidirection,use_stereo_prediction, **kwargs)

    def forward_train(self, left_imgs, right_imgs, **kwargs):
        loss = {}
        assert len(left_imgs.shape) == 5 # B T C H W
        B, T, C, H, W = left_imgs.shape
        #bt_left_imgs = left_imgs.reshape(-1, *left_imgs.shape[2:])
        #bt_right_imgs = right_imgs.reshape(-1, *right_imgs.shape[2:])
        if T==2:
            ref_left_imgs, cur_left_imgs = left_imgs[:,0], left_imgs[:,1]
            ref_right_imgs, cur_right_imgs = right_imgs[:,0], right_imgs[:,1]
            ref_disps = self.depth_net(ref_left_imgs, ref_right_imgs)
            
            
            if not isinstance(ref_disps, (list, tuple)): ref_disps = [ref_disps]


            for level, ref_disp in enumerate(ref_disps): 
                ref_depth = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (ref_disp+1e-6)
                smooth_losses = {f"lv{level}_smooth": self.smooth_loss(ref_disp, ref_left_imgs) }
                loss.update(smooth_losses)
                ## stereo maching 
                SMloss = self.stereo_head(source_imgs=ref_right_imgs, target_disps=ref_disp, target_imgs=ref_left_imgs, direction="r2l")
                SMloss = {f'lv{level}'+k+'_ref':v for k,v in SMloss.items()}
                loss.update(SMloss)
                
        
                
        if T==3: 
            ref_left_imgs, pre_left_imgs, next_left_imgs = left_imgs[:,1], left_imgs[:,0], left_imgs[:,2]
            ref_right_imgs, pre_right_imgs, next_right_imgs = right_imgs[:,1], right_imgs[:,0], right_imgs[:,2]
            ref_disps = self.depth_net(ref_left_imgs, ref_right_imgs) 
            
            if not isinstance(ref_disps, (list, tuple)): ref_disps = [ref_disps]
            for level, ref_disp in enumerate(ref_disps): 
                ref_depth = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (ref_disp+1e-6)
                smooth_losses = {f"lv{level}_smooth": self.smooth_loss(ref_disp, ref_left_imgs) }
                loss.update(smooth_losses)
                ## stereo maching 
                SMloss = self.stereo_head(source_imgs=ref_right_imgs, target_disps=ref_disp, target_imgs=ref_left_imgs, direction="r2l")
                SMloss = {f'lv{level}'+k+'_ref':v for k,v in SMloss.items()}
                loss.update(SMloss)
      

        #loss = {k: v[v>0] for k, v in loss.items()} # mask the zero loss, do not average zero positions
        return loss

    
 
 
    def _do_test(self, left_imgs, right_imgs=None, **kwargs):
        assert len(left_imgs.shape) == 5 # B T C H W
        B, T, C, H, W = left_imgs.shape
        #bt_left_imgs = left_imgs.reshape(-1, *left_imgs.shape[2:])
        #bt_right_imgs = right_imgs.reshape(-1, *right_imgs.shape[2:])
        ref_left_imgs, cur_left_imgs = left_imgs[:,0], left_imgs[:,1]
        ref_right_imgs, cur_right_imgs = right_imgs[:,0], right_imgs[:,1]

        outputs = [[],[], # depth 
                   [],[], # mask 
                    [],[],] # pose
        if self.depth_net is not None:
            disps = self.depth_net(ref_left_imgs, ref_right_imgs) # only decode the first frame's depth
            if isinstance(disps, (list,tuple)):
                disps = disps[0]
            depths = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (disps+1e-6)
        else: # load saved depth 
            if 'pred_disps' in kwargs.keys(): 
                disps = kwargs['pred_disps']
                depths = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (disps+1e-6)
            if 'pred_depths' in kwargs.keys(): 
                depths = kwargs['pred_depths'] 
                disps = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (depths+1e-6)
        outputs[0] = depths.detach().cpu().numpy()
        outputs[1] = kwargs["left_depths"][:,0].detach().cpu().numpy() if "left_depths" in kwargs.keys() else []

        if self.pose_net is not None:
            initcTr, pose_6d = self.pose_net(left_imgs) # cTr 
        else:  
            if self.save_last_pose is None:
                self.save_last_pose = torch.eye(4).to(left_imgs.device).unsqueeze(0).repeat(B,1,1)
            initcTr = self.save_last_pose
        masks = (depths > 1) & (depths < 100) # mask out invalid depth
        pred_pose = initcTr
        #print(pred_pose)
        #pred_pose = self.ddvo_module( Ir=ref_left_imgs, Zr=depths, Ic=cur_left_imgs, \
        #                              K=kwargs["intrinsics"][:,0], imask=masks, cTr=initcTr)
        pred_pose = self.ddvo_head(source_img=cur_left_imgs, target_depth=depths,
                                        sTt=initcTr, K=kwargs["intrinsics"][:,0],  target_img=ref_left_imgs,
                                        target_mask=masks,
                                        test_mode=True)
        
        outputs[4] = pred_pose.detach().cpu().numpy() if isinstance(pred_pose, torch.Tensor)\
                                     else pred_pose

        if self.pose_net is None: # save the last frame's pose
            self.save_last_pose = pred_pose

        if 'pose' in kwargs.keys(): 
            abs_poses = kwargs['pose'] # B T 4 4 load gt pose
            assert T == 2
            gtcTr = torch.linalg.solve(abs_poses[:,1,:,:], abs_poses[:,0,:,:])
            gtrTc = torch.linalg.solve(abs_poses[:,0,:,:], abs_poses[:,1,:,:])
            outputs[5] = gtcTr.detach().cpu().numpy()  

        return outputs

 