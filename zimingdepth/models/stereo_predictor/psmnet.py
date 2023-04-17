import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings

from ..builder import build_backbone, build_cost_processor, build_disp_predictor

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor
import time

@STEREO_PREDICTOR.register_module()
class PSMNet(BaseStereo):
    """
    Base depth method. 

    """
    def __init__(self, backbone, disp_head, neck=None, predict_format="depth", pretrained=None, **kwargs):
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

        self.data = [] # num of frames, each component is a dict{}
        self.set_num_frames = 1
        self.data = [{} for _ in range(self.set_num_frames)]
        self._MAX_DISP= -1
        self.predict_format=predict_format

    def _init_data(self,left_imgs, right_imgs, **kwargs):
        '''
        description: 
        parameter: {*} Input tensor data (GT depth, pose, camera instrinsics etc. are in kwargs)
        return: {*}
        '''        
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
            #print("left pose >> ", self.data[ti]["cTr"])
        if "right_pose" in kwargs:
            self.cTr_right = kwargs["right_pose"][:,0,:,:].float()
            self.rTc_right = kwargs["right_pose"][:,1,:,:].float()
            for ti in range(self.num_frames):
                self.data[ti]["cTr_right"],self.data[ti]["rTc_right"] = self.cTr_right, self.rTc_right
            #print("right pose >> ", self.data[ti]["cTr_right"])
        if "stereo_pose" in kwargs:
            self.rTl = kwargs["stereo_pose"][:,0,:,:]
            self.lTr = kwargs["stereo_pose"][:,1,:,:]
            for ti in range(self.num_frames):
                self.data[ti]["rTl"],self.data[ti]["lTr"] = self.rTl, self.lTr 
        if 'intrinsics' in kwargs:
            intrinsics = kwargs['intrinsics']
            self.left_K44, self.right_K44 = intrinsics[:,0,:,:].float(), intrinsics[:,1,:,:].float()
            self.left_invK44, self.right_invK44 = torch.linalg.inv(self.left_K44.cpu()).cuda(), torch.linalg.inv(self.right_K44.cpu()).cuda()
            for ti in range(self.num_frames):
                self.data[ti]["left_K44"],self.data[ti]["right_K44"] =self.left_K44, self.right_K44
                self.data[ti]["left_invK44"], self.data[ti]["right_invK44"] = self.left_invK44, self.right_invK44 

    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        self._init_data(left_imgs, right_imgs, **kwargs)
        losses = dict()
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        #if self.rescale_img is not None:
        #    leftImage = F.interpolate(leftImage, scale_factor=self.rescale_img, mode='bilinear', align_corners=False)
        #    rightImage = F.interpolate(rightImage, scale_factor=self.rescale_img, mode='bilinear', align_corners=False)
        results = self.extract_disp(leftImage, rightImage)
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
            #        print(self._MAX_DISP)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_left_gt_disp = self.disp_head.loss(results["disps"], gt_disp)
            losses.update(loss_left_gt_disp)
        if "right_gt_disp" in self.data[0].keys() and results["right_disps"] is not None:
            gt_disp = torch.stack([self.data[ti]["right_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_right_gt_disp = self.disp_head.loss(results["right_disps"], gt_disp)
            losses.update(loss_right_gt_disp)
        """ 
        warp_right_img = self.generate_image_right(leftImage, gt_disp)
        vis = "/home/ziliu/vis/check_sceneflow"
        vis_img_tensor(rightImage, vis, "gt_right")
        vis_img_tensor(warp_right_img,vis, "warp_right")
        vis_img_tensor(leftImage,vis, "gt_left")
        vis_depth_tensor(gt_disp, vis,"left_disp")
        vis_img_tensor(torch.abs(warp_right_img-rightImage), vis,"difference")
        print(max(gt_disp.reshape(-1)))
        if max(gt_disp.reshape(-1))> 256:
            exit()
        """
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

        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir
        network_pred_right_disparities = None
        #print("estimate depth in test stage...")
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        t0=time.time()
        results = self.extract_disp(leftImage, rightImage)
        pred_disparities = results["disps"][0] # keep the last prediction
        network_pred_right_disparities = results["right_disps"][0] if results["right_disps"] is not None else None
        #pred_disparities = [pred_disparities.reshape(pred_disparities.shape[0], 1, self.height, self.width)]
        pred_disparities = [pred_disparities]
        t1 = time.time()
        print("time for disparity estimation: ", t1-t0)
        #print(f"rank {rank} >> pred disp  {pred_disparities[0].shape}")
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
        


