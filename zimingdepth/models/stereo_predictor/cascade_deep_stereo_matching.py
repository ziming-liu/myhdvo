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
from .base_ import BaseDepth

@STEREO_PREDICTOR.register_module()
class CascadeDeepStereoMatching(BaseDepth):
    """
    Base depth method. 

    """
    def __init__(self, max_disp, backbone, cost_processor, disp_predictor, num_views=1, batch_norm=True, pretrained=None, disp_l1_loss=None, num_frames=1, **kwargs):
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
        super(CascadeDeepStereoMatching, self).__init__(max_disp, backbone, cost_processor, disp_predictor, 
                num_views, batch_norm, pretrained, disp_l1_loss, **kwargs)

        self.data = [] # num of frames, each component is a dict{}
        self.set_num_frames = num_frames
        self.data = [{} for _ in range(self.set_num_frames)]

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
        self.focal, self.baseline = kwargs["focal"], kwargs["baseline"]
        self.disp2depth_factor = kwargs["focal"] * kwargs["baseline"]
        
        for ti in range(self.num_frames):
            self.data[ti].update({'left_img': left_imgs[:,ti,:,:,:], 
                            'right_img': right_imgs[:,ti,:,:,:]})
            self.data[ti]["disp2depth_factor"] = self.disp2depth_factor # for a same sequence, have same focal,baseline, so only data[0] saves this data
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
                self.data[ti].update({'left_gt_depth': self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['left_gt_disp'] } )
        if 'right_disps' in kwargs:  
            gt_right_disps = kwargs['right_disps'].reshape((self.bs, self.num_frames, 1, self.height,self.width)) 
            for ti in range(self.num_frames):
                self.data[ti].update({  'right_gt_disp': gt_right_disps[:,ti,:,:,:].float()  })
                self.data[ti].update({'right_gt_depth': self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['right_gt_disp'] } )

        if 'left_pred_depths' in kwargs:
            left_pred_depths = kwargs['left_pred_depths'].reshape((self.bs, self.num_frames, 1, self.height,self.width))
            for ti in range(self.num_frames):
                self.data[ti].update({'left_pred_depth': [left_pred_depths[:,ti,:,:,:].float()] })
                self.data[ti].update({'left_pred_disp': [self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['left_pred_depth'][0] ] } )
            self.use_existing_pred_depth = True
        if 'right_pred_depths' in kwargs:  
            right_pred_depths = kwargs['right_pred_depths'].reshape((self.bs, self.num_frames, 1, self.height,self.width)) 
            for ti in range(self.num_frames):
                self.data[ti].update({'right_pred_depth': [right_pred_depths[:,ti,:,:,:].float() ] })
                self.data[ti].update({'right_pred_disp': [self.disp2depth_factor.reshape(self.bs,1,1,1) / self.data[ti]['right_pred_depth'][0] ] } )
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

    def extract_disp(self, left_img, right_img):
        # extract image feature
        ref_fms, tgt_fms = self.backbone(left_img, right_img)
        # compute cost volume, return a list
        costs = self.cost_processor(ref_fms, tgt_fms)
        
        # disparity prediction, multi levels
        disps = [self.disp_predictor(cost) for cost in costs]
        #if len(disps)>1: disps = [torch.mean( torch.stack(disps), 0)] # avg 3 levels
        
        disps_right = None
        if self.num_views==2:
             # compute cost volume, return a list
            costs = self.right_cost_processor( ref_fms, tgt_fms,)
            
            # disparity prediction, multi levels
            disps_right = [self.disp_predictor(cost) for cost in costs]
            #if len(disps_right)>1: disps_right = [torch.mean( torch.stack(disps_right), 0)] # avg 3 levels
        results = dict(
                disps=disps,
                right_disps =disps_right,
                costs=costs,
            )
        return results
 
 

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
        if "left_gt_disp" in self.data[0]:
            gt_disp = torch.stack([self.data[ti]["left_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_left_gt_disp = self.loss(results["disps"], gt_disp)
            losses.update(loss_left_gt_disp)
        if "right_gt_disp" in self.data[0] and results["right_disps"] is not None:
            gt_disp = torch.stack([self.data[ti]["right_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_right_gt_disp = self.loss(results["right_disps"], gt_disp)
            losses.update(loss_right_gt_disp)

        return losses

    def forward_test(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        
        frame_dir = [item['frame_dir'] for item in kwargs['img_metas']]

        return self._do_test( left_imgs, right_imgs, frame_dir,**kwargs)
        
    def _do_test(self, left_imgs, right_imgs,frame_dir,**kwargs):
        """Defines the computation performed at every call when training."""
        self._init_data(left_imgs, right_imgs, **kwargs)
        #from mmcv.runner import get_dist_info, init_dist, load_checkpoint
        #rank, _ = get_dist_info()

        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir
        network_pred_right_disparities = None
        #print("estimate depth in test stage...")
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        results = self.extract_disp(leftImage, rightImage)
        pred_disparities = results["disps"][0] # keep the last prediction
        network_pred_right_disparities = results["right_disps"][0] if results["right_disps"] is not None else None
        pred_disparities = [pred_disparities.reshape(pred_disparities.shape[0], 1, self.height, self.width)]
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
                pred_disparities[j] = F.interpolate(pred_disparities[j], size=(self.height,self.width),mode="bilinear")
                left_pred_depth = torch.stack([self.disp2depth_factor[bi]/(pred_disparities[j].reshape((self.bs,self.num_frames,1,self.height,self.width))[bi,ti,:,:,:]) for bi in range(self.bs)])
                self.data[ti]["left_pred_depth"].append(left_pred_depth)
                left_pred_disp = pred_disparities[j].reshape((self.bs,self.num_frames,1,self.height,self.width))[:,ti,:,:,:] 
                self.data[ti]["left_pred_disp"].append(left_pred_disp)
                if network_pred_right_disparities is not None: # network output right disparity
                    network_pred_right_disparities[j] = F.interpolate(network_pred_right_disparities[j], size=(self.height,self.width),mode="bilinear")
                    pred_right_depth_ti = torch.stack([self.disp2depth_factor[bi]/(network_pred_right_disparities[j].reshape((self.bs,self.num_frames,1,self.height,self.width))[bi,ti,:,:,:]) for bi in range(self.bs)])
                    self.data[ti]["right_pred_depth"].append(pred_right_depth_ti)
                    right_pred_disp = network_pred_right_disparities[j].reshape((self.bs,self.num_frames,1,self.height,self.width))[:,ti,:,:,:] 
                    self.data[ti]["right_pred_disp"].append(right_pred_disp)

        if network_pred_right_disparities is not None or \
                "right_pred_depth" in self.data[0].keys() and len(self.data[0]["right_pred_depth"])!=0:         
            all_pred_depthes = torch.stack([self.data[0]["left_pred_depth"][0].squeeze(1), \
                        self.data[0]["right_pred_depth"][0].squeeze(1) ], 1).float().cpu().numpy()
        else:
            all_pred_depthes = self.data[0]["left_pred_depth"][0].squeeze(1).unsqueeze(1).float().cpu().numpy()
        outputs[0]=all_pred_depthes # batch, views, h, w
        
        #all_gt_depth = all_pred_depthes
        if "left_gt_depth" in self.data[0]:
            if not self.fast_depth:
                all_gt_depth = self.data[0]["left_gt_depth"].squeeze(1).unsqueeze(1).float().cpu().numpy() # batch, views, h, w
            else:
                t0_gt = self.data[0]["left_gt_depth"].squeeze(1)
                t1_gt = self.data[1]["left_gt_depth"].squeeze(1)
                left_gt = torch.stack([t0_gt, t1_gt], 1).reshape((-1,)+t0_gt.shape[-2:])
                all_gt_depth = left_gt.unsqueeze(1).float().cpu().numpy()
            #if "right_gt_depth" in self.data[0]:
            #    all_gt_depth = torch.stack([self.data[0]["left_gt_depth"].squeeze(1), self.data[0]["right_gt_depth"].squeeze(1)]).permute(1,0,2,3).float().cpu().numpy() # batch, views, h, w
            outputs[1] = all_gt_depth
        
        return outputs
        
    def loss(self, pred, gt, **kwargs):
        losses = {}
        if self.disp_l1_loss:
            losses.update(self.disp_l1_loss_func(pred, gt))

        return losses


