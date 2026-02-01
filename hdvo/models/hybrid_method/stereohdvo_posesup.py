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
import time 
from hdvo.models.utils.stereo_warping import *
from hdvo.models.utils.temporal_warping import *
import os

@HYBRID_METHOD.register_module()
class StereoHDVOPosesup(HDVO):
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
                 pretrain=None,
                 vkitti2_flag=False,
                 **kwargs):
        super(StereoHDVOPosesup, self).__init__(depth_net, segment_net, pose_net, brightness_net, 
                                       occ_mask, homo_mask, stereo_head, ddvo_head, smooth_loss, ddvo_module, 
                                       use_stereo_depth, use_TTT, use_sup_pose, bidirection,use_stereo_prediction, **kwargs)

        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)
        self.save_last_pose = None
        self.vkitti2_flag = vkitti2_flag
        self.vis_id = 0
        if pretrain is not None:
            self.__init_weights(pretrain)

    def __init_weights(self, pretrain):
        load_checkpoint(self, pretrain, map_location='cpu')

    def forward_train(self, left_imgs, right_imgs, **kwargs):
        loss = {}
        assert len(left_imgs.shape) == 5 # B T C H W
        B, T, C, H, W = left_imgs.shape
        #bt_left_imgs = left_imgs.reshap
        B, T, C, H, W = left_imgs.shape
        if self.homo_mask is not None:
            left_frame_paths = kwargs['img_metas'][0]["left_frame_paths"]
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
                
                abs_poses = kwargs['pose']
                initcTr = (torch.linalg.inv(abs_poses[:,1,:,:].double()) @ abs_poses[:,0,:,:].double()).float()
                    
                masks = None
                ddvoloss = self.ddvo_head(source_img=cur_left_imgs, target_depth=ref_depth,
                                        sTt=initcTr, K=kwargs["intrinsics"][:,0], target_mask=masks,  target_img=ref_left_imgs)
                ddvoloss = {f'lv{level}'+k:v for k,v in ddvoloss.items()}
                loss.update(ddvoloss)

                
                
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

                abs_poses = kwargs['pose']
                initcTr = (torch.linalg.inv(abs_poses[:,0,:,:].double()) @ abs_poses[:,1,:,:].double()).float()

                masks = None
                ddvoloss_pre = self.ddvo_head(source_img=pre_left_imgs, target_depth=ref_depth,
                                        sTt=initcTr, K=kwargs["intrinsics"][:,0], target_mask=masks,  target_img=ref_left_imgs)
                ddvoloss_pre = {f'lv{level}'+k+'_pre':v for k,v in ddvoloss_pre.items()}
                loss.update(ddvoloss_pre)
                
                initcTr = (torch.linalg.inv(abs_poses[:,2,:,:].double()) @ abs_poses[:,1,:,:].double()).float()
                ddvoloss_next = self.ddvo_head(source_img=next_left_imgs, target_depth=ref_depth,
                                        sTt=initcTr, K=kwargs["intrinsics"][:,0], target_mask=masks, target_img=ref_left_imgs)
                ddvoloss_next = {f'lv{level}'+k+'_next':v for k,v in ddvoloss_next.items()}
                loss.update(ddvoloss_next)

                if self.occ_mask is not None:
                    left_frame_paths = kwargs['img_metas'][0]["left_frame_paths"]
                    #print(left_frame_paths)
                    stereo_warped = stereo_warp(ref_right_imgs, ref_disp, 'r2l')
                    stereo_warped = stereo_warp(ref_right_imgs, ref_disp, 'r2l')
                    temporal_warped = temporal_warp_core(next_left_imgs, ref_depth, initcTr, kwargs["intrinsics"][:,0], torch.linalg.inv(kwargs["intrinsics"][:,0].double()).float(), 'bilinear',)
                    stc_mask = self.occ_mask(stereo_warped, temporal_warped, left_frame_paths)
        loss = {k: v[v>0] for k, v in loss.items()} # mask the zero loss, do not average zero positions
        return loss

    
 
 
    def _do_test(self, left_imgs, right_imgs=None, **kwargs):
        assert len(left_imgs.shape) == 5 # B T C H W
        B, T, C, H, W = left_imgs.shape
        #bt_left_imgs = left_imgs.reshape(-1, *left_imgs.shape[2:])
        #bt_right_imgs = right_imgs.reshape(-1, *right_imgs.shape[2:])
        ref_left_imgs, cur_left_imgs = left_imgs[:,0], left_imgs[:,1]
        ref_right_imgs, cur_right_imgs = right_imgs[:,0], right_imgs[:,1]

        t0 = time.time()

        outputs = [[],[], # depth 
                   [],[], # mask 
                    [],[],] # pose
        if self.depth_net is not None:
            disps = self.depth_net(ref_left_imgs, ref_right_imgs) # only decode the first frame's depth
            if isinstance(disps, (list,tuple)):
                disps = disps[0]
            depths = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (disps+1e-6)
        else:
            if 'pred_disps' in kwargs.keys(): 
                disps = kwargs['pred_disps']
                depths = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (disps+1e-6)
            if 'pred_depths' in kwargs.keys(): 
                depths = kwargs['pred_depths'] 
                disps = (kwargs['baseline']*kwargs['focal']).reshape(-1,1,1,1) / (depths+1e-6)
        outputs[0] = depths.detach().cpu().numpy()
        outputs[1] = kwargs["left_depths"][:,0].detach().cpu().numpy() if "left_depths" in kwargs.keys() else []

        if self.pose_net is not None:
            initcTr, pose_6d = self.pose_net(left_imgs)
        else:  
            if self.save_last_pose is None:
                abs_poses = kwargs['pose']
                self.save_last_pose = torch.linalg.solve(abs_poses[:,1,:,:], abs_poses[:,0,:,:])
            initcTr = self.save_last_pose
        
        t_vo = time.time()
        depths = F.interpolate(depths, scale_factor=1/2, mode="bilinear")
        cur_left_imgs = F.interpolate(cur_left_imgs, scale_factor=1/2, mode="bilinear")
        ref_left_imgs = F.interpolate(ref_left_imgs, scale_factor=1/2, mode="bilinear")
        masks = (depths > 1) & (depths < 100)
        intrinsics = kwargs["intrinsics"][:,0]
        intrinsics[:,0,0], intrinsics[:,0,2], intrinsics[:,1,1], intrinsics[:,1,2] = 0.5*intrinsics[:,0,0], 0.5*intrinsics[:,0,2], 0.5*intrinsics[:,1,1], 0.5*intrinsics[:,1,2]
        
        pred_pose = self.ddvo_head(source_img=cur_left_imgs, target_depth=depths,
                                        sTt=initcTr, K=intrinsics,  target_img=ref_left_imgs,
                                        target_mask=masks,
                                        test_mode=True)
        self.save_last_pose = pred_pose

        torch.cuda.synchronize()
        t1 = time.time()
        self.timer["sum_time"] += t1 - t0
        self.timer["count"] += 1
        if self.timer["f0_time"] == 0:
            self.timer["f0_time"] = t1 - t0
        else:
            self.timer["avg_time"] = (self.timer["sum_time"]-self.timer["f0_time"]) / (self.timer["count"]-1)
            self.timer["fps"] = 1 / self.timer["avg_time"]

        outputs[4] = pred_pose.detach().cpu().numpy() if isinstance(pred_pose, torch.Tensor) else pred_pose

        if self.pose_net is None:
            self.save_last_pose = pred_pose

        if 'pose' in kwargs.keys(): 
            abs_poses = kwargs['pose'] # B T 4 4 load gt pose
            assert T == 2
            gtcTr = torch.linalg.solve(abs_poses[:,1,:,:], abs_poses[:,0,:,:])
            gtrTc = torch.linalg.solve(abs_poses[:,0,:,:], abs_poses[:,1,:,:])
            outputs[5] = gtcTr.detach().cpu().numpy()  

            assert T == 2
            gtcTr = torch.linalg.solve(abs_poses[:,1,:,:], abs_poses[:,0,:,:])
            gtrTc = torch.linalg.solve(abs_poses[:,0,:,:], abs_poses[:,1,:,:])
            outputs[5] = gtcTr.detach().cpu().numpy()  

        return outputs