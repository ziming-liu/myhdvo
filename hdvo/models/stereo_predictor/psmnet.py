import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings

from ..builder import build_backbone, build_cost_processor, build_disp_predictor, build_loss

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor
import time
from ..utils.temporal_warping import temporal_warp_c2r,  temporal_warp_r2c, temporal_warp_core
from ..utils.stereo_warping import stereo_warp_r2l, stereo_warp_l2r

@STEREO_PREDICTOR.register_module()
class PSMNet(BaseStereo):
    """
    Base depth method. 

    """
    def __init__(self, backbone, disp_head, neck=None, predict_format="depth", 
                 pretrained=None, photo_loss=None, struct_loss=None, smooth_loss=None, 
                 grad_hessian=False, cu_grad_noesm=False, cu_grad=False,
                  use_sup_loss=True, use_unsup_loss=False, onlyleft=True,
                  grid_sample_type="pytorch",
                   lam_mask=None, stc_mask=None, **kwargs):
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
        super(PSMNet, self).__init__( backbone, disp_head, neck, pretrained, **kwargs)
        self.cu_grad_noesm = cu_grad_noesm
        self.cu_grad = cu_grad
        self.grad_hessian = grad_hessian
        self.grid_sample_type = grid_sample_type
        self.onlyleft = onlyleft
        self.lam_mask = lam_mask
        self.stc_mask = stc_mask

        self.use_sup_loss = use_sup_loss
        self.use_unsup_loss = use_unsup_loss
        assert use_sup_loss != use_unsup_loss, "sup_loss and unsup_loss cannot be both True or False"

        self.data = [] # num of frames, each component is a dict{}
        self.set_num_frames = 1
        self.data = [{} for _ in range(self.set_num_frames)]
        self._MAX_DISP= -1
        self.predict_format=predict_format
        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)

        self.photo_loss_func = build_loss(photo_loss) if photo_loss is not None else None
        self.struct_loss_func = build_loss(struct_loss) if struct_loss is not None else None
        self.smooth_loss_func = build_loss(smooth_loss) if smooth_loss is not None else None

    def _init_data(self,left_imgs, right_imgs, **kwargs):
        '''
        description: 
        parameter: {*} Input tensor data (GT depth, pose, camera instrinsics etc. are in kwargs)
        return: {*}
        '''        
        self.data = [{} for _ in range(left_imgs.shape[1])]
        left_imgs = left_imgs.reshape((left_imgs.shape[0],-1)+left_imgs.shape[-3:] )
        right_imgs = right_imgs.reshape((right_imgs.shape[0],-1)+right_imgs.shape[-3:] )
        bs, num_frames, channel,  height, width = left_imgs.shape
        self.set_num_frames = num_frames
        #assert self.num_frames == self.set_num_frames, "setted num in configs is not same as input num frames"
        self.bs, self.num_frames, self.channel,  self.height, self.width = bs, num_frames, channel, height, width
        
        # reduce dimension of segments, -> bs, t, c, h, w
        #left_imgs = left_imgs.reshape((-1, ) + left_imgs.shape[2:])
        #right_imgs = right_imgs.reshape((-1, ) + right_imgs.shape[2:])
        if "focal" in kwargs and "baseline" in kwargs:
            self.focal, self.baseline = kwargs["focal"], kwargs["baseline"]
            self.focal_left = self.focal
            self.focal_right = self.focal
            self.disp2depth_factor = kwargs["focal"] * kwargs["baseline"]
            for ti in range(self.num_frames):
                self.data[ti]["disp2depth_factor"] = self.disp2depth_factor # for a same sequence, have same focal,baseline, so only data[0] saves this data
        
        for ti in range(self.num_frames):
            self.data[ti].update({'left_img': left_imgs[:,ti,:,:,:], 
                            'right_img': right_imgs[:,ti,:,:,:]})
        if 'left_depths' in kwargs:
            gt_left_depths = kwargs['left_depths'].reshape((self.bs, self.num_frames, 1, self.height,self.width))
            for ti in range(self.num_frames):
                self.data[ti].update({'left_gt_depth': gt_left_depths[:,ti,:,:,:].float() })
                self.data[ti].update({'left_gt_disp': self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['left_gt_depth'] } )
        if 'right_depths' in kwargs:  
            gt_right_depths = kwargs['right_depths'].reshape((self.bs, self.num_frames, 1, self.height,self.width)) 
            for ti in range(self.num_frames):
                self.data[ti].update({  'right_gt_depth': gt_right_depths[:,ti,:,:,:].float()  })
                self.data[ti].update({'right_gt_disp': self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['right_gt_depth'] } )
        if 'left_disps' in kwargs:
            gt_left_disps = kwargs['left_disps'].reshape((self.bs, self.num_frames, 1, self.height,self.width))
            for ti in range(self.num_frames):
                self.data[ti].update({'left_gt_disp': gt_left_disps[:,ti,:,:,:].float() })
                #self.data[ti].update({'left_gt_depth': self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['left_gt_disp'] } )
        if 'right_disps' in kwargs:  
            gt_right_disps = kwargs['right_disps'].reshape((self.bs, self.num_frames, 1, self.height,self.width)) 
            for ti in range(self.num_frames):
                self.data[ti].update({  'right_gt_disp': gt_right_disps[:,ti,:,:,:].float()  })
                #self.data[ti].update({'right_gt_depth': self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['right_gt_disp'] } )

        if 'left_pred_depths' in kwargs:
            left_pred_depths = kwargs['left_pred_depths'].reshape((self.bs, self.num_frames, 1, self.height,self.width))
            for ti in range(self.num_frames):
                self.data[ti].update({'left_pred_depth': [left_pred_depths[:,ti,:,:,:].float()] })
                #self.data[ti].update({'left_pred_disp': [self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['left_pred_depth'][0] ] } )
            self.use_existing_pred_depth = True
        if 'right_pred_depths' in kwargs:  
            right_pred_depths = kwargs['right_pred_depths'].reshape((self.bs, self.num_frames, 1, self.height,self.width)) 
            for ti in range(self.num_frames):
                self.data[ti].update({'right_pred_depth': [right_pred_depths[:,ti,:,:,:].float() ] })
                #self.data[ti].update({'right_pred_disp': [self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['right_pred_depth'][0] ] } )
            self.use_existing_pred_depth = True
        

        # temporal inputs, using camera poses
        # TODO: now only support load pose, K for 2 frames setting
        if "pose" in kwargs:
            self.cTr = kwargs["pose"][:,0,:,:].float()
            self.rTc = kwargs["pose"][:,1,:,:].float()
            for ti in range(self.num_frames):
                self.data[ti]["cTr"],self.data[ti]["rTc"] = self.cTr, self.rTc 
            ##print("left pose >> ", self.data[ti]["cTr"])
        if "right_pose" in kwargs:
            self.cTr_right = kwargs["right_pose"][:,0,:,:].float()
            self.rTc_right = kwargs["right_pose"][:,1,:,:].float()
            for ti in range(self.num_frames):
                self.data[ti]["cTr_right"],self.data[ti]["rTc_right"] = self.cTr_right, self.rTc_right
            ##print("right pose >> ", self.data[ti]["cTr_right"])
        if "stereo_pose" in kwargs:
            self.rTl = kwargs["stereo_pose"][:,0,:,:]
            self.lTr = kwargs["stereo_pose"][:,1,:,:]
            for ti in range(self.num_frames):
                self.data[ti]["rTl"],self.data[ti]["lTr"] = self.rTl, self.lTr 
        if 'intrinsics' in kwargs:
            intrinsics = kwargs['intrinsics']
            self.left_K44, self.right_K44 = intrinsics[:,0,:,:].float(), intrinsics[:,1,:,:].float()
            self.left_invK44, self.right_invK44 = torch.linalg.inv(self.left_K44.cpu()).cuda(), torch.linalg.inv(self.right_K44.cpu()).cuda()
            self.K = self.left_K44
            for ti in range(self.num_frames):
                self.data[ti]["left_K44"],self.data[ti]["right_K44"] =self.left_K44, self.right_K44
                self.data[ti]["left_invK44"], self.data[ti]["right_invK44"] = self.left_invK44, self.right_invK44 

    def forward_train_subnetwork(self, left_imgs, right_imgs, **kwargs):
        results = self.extract_disp(left_imgs, right_imgs,)
        return results["disps"]


    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        B = left_imgs.shape[0]
        self._init_data(left_imgs, right_imgs, **kwargs)
        
        if 'intrinsics' in kwargs:
            self.K = kwargs['intrinsics'].reshape((B*2,)+kwargs['intrinsics'].shape[-2:]).float()
        if 'focal' in kwargs:
            self.focal_left, self.baseline = kwargs['focal'].reshape(B,1,1,1), kwargs["baseline"].reshape(B,1,1,1)
            self.focal_right =  self.focal_left
        if 'pose' in kwargs:
            self.gt_poses = kwargs["pose"] # B T 4 4

        losses = dict()
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        #if self.rescale_img is not None:
        #    leftImage = F.interpolate(leftImage, scale_factor=self.rescale_img, mode='bilinear', align_corners=False)
        #    rightImage = F.interpolate(rightImage, scale_factor=self.rescale_img, mode='bilinear', align_corners=False)
        results = self.extract_disp(leftImage, rightImage)

        if self.use_sup_loss:
            losses.update(self.sup_loss(results))
        if self.use_unsup_loss:
            for i in range(len(results['disps'])):
                losses.update(self.unsup_loss(results["disps"][i],  left_imgs, right_imgs,\
                                               raw_left_imgs=None, raw_right_imgs=None, level=i))

        #if self.disp_head.stereo_feat_cons_losses is not None:
        #    left_gt_disp = torch.stack([self.data[ti]["left_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
        #    right_gt_disp = torch.stack([self.data[ti]["right_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
        #    loss_rectify = self.disp_head.loss_rectify_calib(results["left_feats"], results["right_feats"],
        #                                                     F.interpolate(left_gt_disp, scale_factor=1/4, mode="bilinear" ),
        #                                                      F.interpolate(right_gt_disp, scale_factor=1/4, mode="bilinear" ))
        #    losses.update(loss_rectify)
        #vis_img_tensor(leftImage, "/home/ziliu/vis/monodepth2", "LEFTIMG")
        #vis_img_tensor(rightImage, "/home/ziliu/vis/monodepth2", "RIGHTIMG")
        #vis_depth_tensor(results["disps"][0],"/home/ziliu/vis/monodepth2", "PRED" )
        #vis_depth_tensor(self.data[0]["left_gt_depth"],"/home/ziliu/vis/monodepth2", "GT" )
        
        """ 
        warp_right_img = self.generate_image_right(leftImage, gt_disp)
        vis = "/home/ziliu/vis/check_sceneflow"
        vis_img_tensor(rightImage, vis, "gt_right")
        vis_img_tensor(warp_right_img,vis, "warp_right")
        vis_img_tensor(leftImage,vis, "gt_left")
        vis_depth_tensor(gt_disp, vis,"left_disp")
        vis_img_tensor(torch.abs(warp_right_img-rightImage), vis,"difference")
        #print(max(gt_disp.reshape(-1)))
        if max(gt_disp.reshape(-1))> 256:
            exit()
        """
        return losses
    def sup_loss(self, results ):
        losses = dict()
        if "left_gt_depth" in self.data[0].keys():
            gt_depth = torch.stack([self.data[ti]["left_gt_depth"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            results_depths = []
            for i in range(len(results["disps"])):
                results_depths.append((self.focal*self.baseline).reshape((-1,1,1,1)).repeat(1,1,results["disps"][i].shape[-2],results["disps"][i].shape[-1]) / (results["disps"][i]+1e-3))
            loss_left_gt_disp = self.disp_head.loss(results_depths, gt_depth)
            losses.update(loss_left_gt_disp)
            return losses
        if "left_gt_disp" in self.data[0].keys():
            gt_disp = torch.stack([self.data[ti]["left_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #if torch.max(gt_disp)>self._MAX_DISP:
            #        self._MAX_DISP =torch.max(gt_disp)
            #        #print(self._MAX_DISP)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_left_gt_disp = self.disp_head.loss(results["disps"], gt_disp)
            losses.update(loss_left_gt_disp)
        if "right_gt_disp" in self.data[0].keys() and results["right_disps"] is not None:
            gt_disp = torch.stack([self.data[ti]["right_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_right_gt_disp = self.disp_head.loss(results["right_disps"], gt_disp)
            losses.update(loss_right_gt_disp)
        return losses
    def unsup_loss(self, disp,  left_imgs, right_imgs, raw_left_imgs=None, raw_right_imgs=None, level=0):
        B,T,C,H,W = left_imgs.shape
        left_imgs_BT, right_imgs_BT = left_imgs.reshape((B*T,C,H,W)), right_imgs.reshape((B*T,C,H,W))
        losses = dict()
        disp = disp.reshape((B,T,1,H,W))
        left_disp_bt = disp[:,:,0,...].reshape((B*T,1,H,W))
        if not self.onlyleft:
            right_disp_bt = disp[:,:,1,...].reshape((B*T,1,H,W))
        smooth_loss_stereo = self.smooth_loss_func(left_disp_bt, left_imgs_BT.detach())
        if not self.onlyleft:
            smooth_loss_stereo = self.smooth_loss_func(right_disp_bt, right_imgs_BT.detach()) + smooth_loss_stereo
        losses.update({f"l{level}_loss_smooth_stereo": smooth_loss_stereo})
        # on left view  >> stereo loss
        stereo_warping_res_leftview = stereo_warp_r2l(right_imgs_BT, left_disp_bt, padding_mode="border", grid_sample_type=self.grid_sample_type, gt_map=left_imgs_BT ) # B*T C H W
        unvalid_mask_leftview = (stereo_warping_res_leftview != 0).all(dim=1, keepdim=True).long()
        photo_loss_stereo_Lview = self.photo_loss_func(stereo_warping_res_leftview, left_imgs_BT)
        struct_loss_stereo_Lview = self.struct_loss_func(stereo_warping_res_leftview, left_imgs_BT)
        #vis_depth_tensor(stereo_warping_res_leftview, "/home/ziliu/vis/", "leftwarping")
        #vis_depth_tensor(left_imgs_BT, "/home/ziliu/vis/",  "leftgt")
        
        # on right view >> stereo loss
        if not self.onlyleft:
            stereo_warping_res_rightview = stereo_warp_l2r(left_imgs_BT, right_disp_bt,  padding_mode="border", grid_sample_type=self.grid_sample_type, gt_map=right_imgs_BT ) # B*T C H W  )
            unvalid_mask_rightview = (stereo_warping_res_rightview != 0).all(dim=1, keepdim=True).long()
            photo_loss_stereo_Rview = self.photo_loss_func(stereo_warping_res_rightview, right_imgs_BT) 
            struct_loss_stereo_Rview = self.struct_loss_func(stereo_warping_res_rightview, right_imgs_BT) 
            #vis_depth_tensor(stereo_warping_res_rightview, "/home/ziliu/vis/", "rightwarping")
            #vis_depth_tensor(right_imgs_BT, "/home/ziliu/vis/",  "rightgt")
        if self.lam_mask is not None:
            lam_mask_leftstereoloss = self.lam_mask(raw_left_imgs.reshape(B*T,C,H,W)).long().detach()
            if not self.onlyleft:
                lam_mask_rightstereoloss = self.lam_mask(raw_right_imgs.reshape(B*T,C,H,W)).long().detach()
            photo_loss_stereo_Lview = photo_loss_stereo_Lview * lam_mask_leftstereoloss.detach()
            struct_loss_stereo_Lview = struct_loss_stereo_Lview * lam_mask_leftstereoloss.detach()
            lam_mask_left = lam_mask_leftstereoloss.reshape(B,T,1,H,W).detach()
            if not self.onlyleft:
                photo_loss_stereo_Rview = photo_loss_stereo_Rview * lam_mask_rightstereoloss.detach()
                struct_loss_stereo_Rview = struct_loss_stereo_Rview * lam_mask_rightstereoloss.detach()
                lam_mask_right = lam_mask_rightstereoloss.reshape(B,T,1,H,W).detach()
            
            

        #assert self.K_left.equal(self.K_right), f"K_left {self.K_left} and K_right {self.K_right} should be the same"
        left_stc_masks = []
        right_stc_masks = []
        #self.K_left = torch.cat([self.K_left, self.K_right], dim=0)
        if not self.onlyleft:
            self.K_left = self.K
            self.focal_left = torch.cat([self.focal_left, self.focal_right], dim=0).reshape(B*2,1,1)
            self.baseline = self.baseline.repeat(2,1,1)
        else:
            self.K_left = self.K[:B,...]
            self.focal_left = self.focal_left.reshape(B,1,1)
            self.baseline = self.baseline.reshape(B,1,1)
        for ref_id, cur_id in zip(range(0,T-1), range(1,T)):
            if not self.onlyleft:
                imgs_reference = torch.cat([left_imgs[:,ref_id,...],right_imgs[:,ref_id,...]], dim=0) #B*2 C H W
                imgs_target = torch.cat([left_imgs[:,cur_id,...],right_imgs[:,cur_id,...]], dim=0) #B*2 C H W
                disps_reference = torch.cat([disp[:,ref_id,0,...],disp[:,ref_id,1,...]], dim=0) #B*2 C H W
                disps_target = torch.cat([disp[:,cur_id,0,...],disp[:,cur_id,1,...]], dim=0) #B*2 C H W
            else:
                imgs_reference = left_imgs[:,ref_id,...]
                imgs_target = left_imgs[:,cur_id,...]
                disps_reference = disp[:,ref_id,0,...]
                disps_target = disp[:,cur_id,0,...]
            # stereo images on reference view >> temporal loss
            depth = (self.focal_left * self.baseline)  / (disps_reference+1e-8)
            cTr = torch.linalg.solve(self.gt_poses[:,cur_id,:,:], self.gt_poses[:,ref_id,:,:])
            #cTr = torch.bmm(torch.linalg.inv(self.gt_poses[:,cur_id,:,:].float().cpu()).cuda(), self.gt_poses[:,ref_id,:,:].float()) # inv(Tc) * Tr
            if not self.onlyleft: cTr = cTr.repeat(2,1,1) # left + right, use same camera poses
            temporal_warping_res_refview = temporal_warp_c2r(imgs_target, depth, cTr, 
                                            self.K_left, torch.linalg.inv(self.K_left.cpu()).cuda(),  padding_mode="border", grid_sample_type=self.grid_sample_type, gt_map=imgs_reference  ) # B*2 C H W
            unvalid_mask_refview = (temporal_warping_res_refview != 0).all(dim=1, keepdim=True).long()
            photo_loss_temporal_refview = self.photo_loss_func(temporal_warping_res_refview, imgs_reference)
            struct_loss_temporal_refview = self.struct_loss_func(temporal_warping_res_refview, imgs_reference)
            #vis_depth_tensor(temporal_warping_res_refview, "/home/ziliu/vis/", "refwarping")
            #vis_depth_tensor(imgs_reference, "/home/ziliu/vis/",  "refgt")
            # stereo image on target view >> temporal loss
            cur_depth = (self.focal_left * self.baseline)  / (disps_target+1e-8)
            rTc = torch.linalg.solve(self.gt_poses[:,ref_id,:,:], self.gt_poses[:,cur_id,:,:])
            #rTc = torch.bmm(torch.linalg.inv(self.gt_poses[:,ref_id,:,:].float().cpu()).cuda(), self.gt_poses[:,cur_id,:,:].float()) # inv(Tr) * Tc
            if not self.onlyleft: rTc = rTc.repeat(2,1,1)
            temporal_warping_res_curview = temporal_warp_r2c(imgs_reference, cur_depth, rTc,
                                            self.K_left, torch.linalg.inv(self.K_left.cpu()).cuda(), padding_mode="border", grid_sample_type=self.grid_sample_type, gt_map=imgs_target    )
            unvalid_mask_curview = (temporal_warping_res_curview != 0).all(dim=1, keepdim=True).long()
            photo_loss_temporal_curview = self.photo_loss_func(temporal_warping_res_curview, imgs_target)  
            struct_loss_temporal_curview = self.struct_loss_func(temporal_warping_res_curview, imgs_target) 
            #vis_depth_tensor(temporal_warping_res_curview, "/home/ziliu/vis/", "curwarping")
            #vis_depth_tensor(imgs_target, "/home/ziliu/vis/",  "curgt")
            
            if self.lam_mask is not None:
                lam_mask_reftemloss = torch.cat([lam_mask_left[:,ref_id,...],lam_mask_right[:,ref_id,...]], dim=0).detach()
                photo_loss_temporal_refview *= lam_mask_reftemloss
                lam_mask_curtemloss = torch.cat([lam_mask_left[:,cur_id,...],lam_mask_right[:,cur_id,...]], dim=0).detach()
                photo_loss_temporal_curview *= lam_mask_curtemloss
            if self.stc_mask is not None:
                """
                   left right
                t0  00   01
                t1  10   11 
                """
                stc_mask_00 = self.stc_mask( stereo_warping_res_leftview.reshape(B,T,C,H,W)[:,ref_id], 
                                            temporal_warping_res_refview.reshape(B,2,C,H,W)[:,0],).long()
                
                stc_mask_10 = self.stc_mask( stereo_warping_res_leftview.reshape(B,T,C,H,W)[:,cur_id],
                                            temporal_warping_res_curview.reshape(B,2,C,H,W)[:,0],).long()
                stc_mask_01 = self.stc_mask( stereo_warping_res_rightview.reshape(B,T,C,H,W)[:,ref_id],
                                            temporal_warping_res_refview.reshape(B,2,C,H,W)[:,1],).long()
                stc_mask_11 = self.stc_mask( stereo_warping_res_rightview.reshape(B,T,C,H,W)[:,cur_id],
                                            temporal_warping_res_curview.reshape(B,2,C,H,W)[:,1],).long()
                stc_mask_reftemloss = torch.cat([stc_mask_00,stc_mask_01], dim=0).detach()
                photo_loss_temporal_refview *= stc_mask_reftemloss
                stc_mask_curtemloss = torch.cat([stc_mask_10,stc_mask_11], dim=0).detach()
                photo_loss_temporal_curview *= stc_mask_curtemloss
                struct_loss_temporal_refview *= stc_mask_reftemloss
                struct_loss_temporal_curview *= stc_mask_curtemloss
                # save stc masks for stereo losses
                if len(left_stc_masks) == 0:
                    left_stc_masks.append( stc_mask_00)
                left_stc_masks.append( stc_mask_10)
                if len(right_stc_masks) == 0:
                    right_stc_masks.append( stc_mask_01)
                right_stc_masks.append( stc_mask_11)
            
            mask_idx_tem_ref = torch.ones_like(photo_loss_temporal_refview).long()# * unvalid_mask_refview
            mask_idx_tem_cur = torch.ones_like(photo_loss_temporal_curview).long()# * unvalid_mask_curview
            if self.lam_mask is not None:
                mask_idx_tem_ref *= lam_mask_reftemloss.long()
                mask_idx_tem_cur *= lam_mask_curtemloss.long()
            if self.stc_mask is not None:
                mask_idx_tem_ref *= stc_mask_reftemloss.long()
                mask_idx_tem_cur *= stc_mask_curtemloss.long()
            mask_idx_tem_ref, mask_idx_tem_cur = mask_idx_tem_ref.bool().detach().clone(), mask_idx_tem_cur.bool().detach().clone()
            losses.update({f"l{level}_t0_loss_photo_temporal_f{ref_id},f{cur_id}": photo_loss_temporal_refview[mask_idx_tem_ref]})
            losses.update({f"l{level}_t1_loss_photo_temporal_f{ref_id},f{cur_id}": photo_loss_temporal_curview[mask_idx_tem_cur]})
            losses.update({f"l{level}_t0_loss_struct_temporal_f{ref_id},f{cur_id}": struct_loss_temporal_refview[mask_idx_tem_ref]})
            losses.update({f"l{level}_t1_loss_struct_temporal_f{ref_id},f{cur_id}": struct_loss_temporal_curview[mask_idx_tem_cur]})

        ########### end for temporal loss ###########
        if self.stc_mask is not None:
            # stereo loss 
            stc_mask_leftstereoloss = torch.cat(left_stc_masks, dim=0).detach()
            photo_loss_stereo_Lview *= stc_mask_leftstereoloss
            stc_mask_rightstereoloss = torch.cat(right_stc_masks, dim=0).detach()
            photo_loss_stereo_Rview *= stc_mask_rightstereoloss
            struct_loss_stereo_Lview *= stc_mask_leftstereoloss
            struct_loss_stereo_Rview *=  stc_mask_rightstereoloss
        mask_idx_stereoleft = torch.ones_like(photo_loss_stereo_Lview).long()# * unvalid_mask_leftview
        if not self.onlyleft: mask_idx_stereoright = torch.ones_like(photo_loss_stereo_Rview).long().detach() #* unvalid_mask_rightview
        if self.lam_mask is not None:
            mask_idx_stereoleft *= lam_mask_leftstereoloss
            if not self.onlyleft: mask_idx_stereoright *= lam_mask_rightstereoloss
        if self.stc_mask is not None:
            mask_idx_stereoleft *= stc_mask_leftstereoloss
            if not self.onlyleft: mask_idx_stereoright *= stc_mask_rightstereoloss
        mask_idx_stereoleft = mask_idx_stereoleft.bool().detach().clone()
        if not self.onlyleft: mask_idx_stereoright = mask_idx_stereoright.bool().detach().clone()
        losses.update({f"l{level}_L_loss_photo_stereo": photo_loss_stereo_Lview[mask_idx_stereoleft]})
        if not self.onlyleft: losses.update({f"l{level}_R_loss_photo_stereo": photo_loss_stereo_Rview[mask_idx_stereoright]})
        losses.update({f"l{level}_L_loss_struct_stereo": struct_loss_stereo_Lview[mask_idx_stereoleft]})
        if not self.onlyleft: losses.update({f"l{level}_R_loss_struct_stereo": struct_loss_stereo_Rview[mask_idx_stereoright]})

        return losses
    
    def smooth_loss(self, disp, img):
        losses = {}
        if not isinstance(disp, (list, tuple)):
            disp = [disp] 
        for i in range(len(disp)):
            losses[f"smooth_loss_{i}"] = self.smooth_loss_func(disp[i], img)
        return losses

    def generate_image_left(self, img, disp,mod="dense"):
        return self.apply_disparity(img, -disp,mod, name="left_mask") # from left view to right view projection, x_base-left_disparity
    
    def generate_image_right(self, img, disp,mod="dense"):
        return self.apply_disparity(img,  disp,mod, name="right_mask") 
    
    
    def apply_disparity(self, img, disp,mod="dense",name=None):
        batch_size, _, height, width = img.size()
        y_base, x_base = torch.meshgrid(
                                torch.linspace(0.0 , height - 1.0 , height),
                                torch.linspace(0.0,   width - 1.0,  width))
        y_base, x_base =  y_base.unsqueeze(0).repeat(batch_size,1,1).to(img.device), \
                         x_base.unsqueeze(0).repeat(batch_size,1,1).to(img.device)
        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # norm for grid_sample 
        flow_field[...,0] /=(width-1)
        flow_field[...,1] /=(height-1)
        flow_field = (flow_field-0.5)*2.0
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, flow_field, mode='bilinear',padding_mode="zeros",align_corners=False)

        return output#,visible_mask

    def forward_test(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        

        return self._do_test( left_imgs, right_imgs,**kwargs)
        
    def _do_test(self, left_imgs, right_imgs,**kwargs):
        """Defines the computation performed at every call when training."""
        self._init_data(left_imgs, right_imgs, **kwargs)
        #from mmcv.runner import get_dist_info, init_dist, load_checkpoint
        #rank, _ = get_dist_info()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        t0 = time.time()
        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir
        network_pred_right_disparities = None
        ##print("estimate depth in test stage...")
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        results = self.extract_disp(leftImage, rightImage)
        end.record()
        torch.cuda.synchronize()

        print("torch time >> ", start.elapsed_time(end))

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
        #print("time for disparity estimation: ", t1-t0)
        pred_disparities = results["disps"][0] # keep the last prediction
        network_pred_right_disparities = results["right_disps"][0] if results["right_disps"] is not None else None
        #pred_disparities = [pred_disparities.reshape(pred_disparities.shape[0], 1, self.height, self.width)]
        pred_disparities = [pred_disparities]
        
        ##print(f"rank {rank} >> pred disp  {pred_disparities[0].shape}")
        self.num_level = len(pred_disparities)
        for ti in range(self.num_frames):
            pred_depth_ti = []
            pred_right_depth_ti = []
            self.data[ti]["left_pred_depth"] = []
            self.data[ti]["left_pred_disp"] = []
            self.data[ti]["right_pred_depth"] = []
            self.data[ti]["right_pred_disp"] = []
            self.data[ti]["num_level"] = self.num_level
            for j in range(self.num_level):
                pred_H, pred_W = pred_disparities[j].shape[-2:]
                #pred_disparities[j] = (self.width/pred_W) *  F.interpolate(pred_disparities[j], size=(self.height,self.width),mode="bilinear")
                if "focal" in kwargs:
                    left_pred_depth = (pred_W/self.width) * torch.stack([self.disp2depth_factor[bi]/(1e-10+pred_disparities[j].reshape((self.bs,self.num_frames,1,pred_H,pred_W))[bi,ti,:,:,:]) for bi in range(self.bs)])
                else:
                    left_pred_depth = pred_disparities[j]
                self.data[ti]["left_pred_depth"].append(left_pred_depth)
                left_pred_disp = pred_disparities[j].reshape((self.bs,self.num_frames,1, pred_H, pred_W))[:,ti,:,:,:] 
                self.data[ti]["left_pred_disp"].append(left_pred_disp)
                if network_pred_right_disparities is not None: # network output right disparity
                    #network_pred_right_disparities[j] =  (self.width/pred_W) *  F.interpolate(network_pred_right_disparities[j], size=(self.height,self.width),mode="bilinear")
                    if "focal" in kwargs:
                        pred_right_depth_ti = (pred_W/self.width) * torch.stack([self.disp2depth_factor[bi]/(1e-10+network_pred_right_disparities[j].reshape((self.bs,self.num_frames,1,pred_H,pred_W))[bi,ti,:,:,:]) for bi in range(self.bs)])
                    else:
                        pred_right_depth_ti = network_pred_right_disparities[j]
                    self.data[ti]["right_pred_depth"].append(pred_right_depth_ti)
                    right_pred_disp = network_pred_right_disparities[j].reshape((self.bs,self.num_frames,1,pred_H, pred_W))[:,ti,:,:,:] 
                    self.data[ti]["right_pred_disp"].append(right_pred_disp)

        if self.predict_format == "depth":
            if network_pred_right_disparities is not None or \
                    "right_pred_depth" in self.data[0].keys() and len(self.data[0]["right_pred_depth"])!=0:         
                all_pred_depthes = torch.stack([self.data[0]["left_pred_depth"][0].squeeze(1), \
                            self.data[0]["right_pred_depth"][0].squeeze(1) ], 1).float().cpu().numpy()
            else:
                all_pred_depthes = self.data[0]["left_pred_depth"][0].squeeze(1).unsqueeze(1).float().cpu().numpy()
            outputs[0]=all_pred_depthes # batch, views, h, w
        elif self.predict_format == "disparity":
            if network_pred_right_disparities is not None or \
                    "right_pred_disp" in self.data[0].keys() and len(self.data[0]["right_pred_disp"])!=0:         
                predisps = torch.stack([self.data[0]["left_pred_disp"][0].squeeze(1), \
                            self.data[0]["right_pred_disp"][0].squeeze(1) ], 1).float().cpu().numpy()
            else:
                predisps = self.data[0]["left_pred_disp"][0].squeeze(1).unsqueeze(1).float().cpu().numpy()
            outputs[0]=predisps # batch, views, h, w
        else:
            raise ValueError
        
        if "left_gt_disp" in self.data[0].keys():
            gtdisp = self.data[0]["left_gt_disp"].squeeze(1).unsqueeze(1).float().cpu().numpy() # batch, views, h, w
            if "right_gt_disp" in self.data[0].keys():
                gtdisp = torch.cat([self.data[0]["left_gt_disp"].squeeze(1).unsqueeze(1),  self.data[0]["right_gt_disp"].squeeze(1).unsqueeze(1) ], 1).float().cpu().numpy()
            outputs[1] = gtdisp

        
        if "left_gt_depth" in self.data[0].keys():
            all_gt_depth = self.data[0]["left_gt_depth"].squeeze(1).unsqueeze(1).float().cpu().numpy() # batch, views, h, w
            outputs[1] = all_gt_depth
        
        return outputs
        


