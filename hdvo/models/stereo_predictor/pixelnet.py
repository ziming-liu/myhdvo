import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings
import torch.distributed as dist
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from ..builder import build_backbone, build_neck, build_disp_predictor,build_loss

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor
from ..utils.inverse_warp_3d import inverse_warp_3d
import time 

@STEREO_PREDICTOR.register_module()
class PixelNet(nn.Module):
    """
    Base depth method. 

    """
    def __init__(self, backbone, losses, out_channels=256, neck=None, 
                 typeIV="nomask", alpha=1, normalize=True, sample_rate=4,
                   max_disp=192, predict_format="depth", pretrained=None, 
                   use_conv_prediction=False, use_fast_prediction=False,
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
        super(PixelNet, self).__init__()
        self.subnetwork = subnetwork
        self.typeIV = typeIV
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.normalize = normalize
        self.data = [] # num of frames, each component is a dict{}
        self.set_num_frames = 1
        self.data = [{} for _ in range(self.set_num_frames)]
        self._MAX_DISP= -1
        self.max_disp = max_disp
        self.predict_format=predict_format
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        self.disp_loss_func = build_loss(losses)
        self.mask_template = None
        self.indices= None
        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)
        self.classify = nn.Conv3d(out_channels, 1, kernel_size=1, padding=0)

        self.use_conv_prediction = use_conv_prediction
        self.use_fast_prediction = use_fast_prediction
        if use_conv_prediction:
            disp_sample = torch.linspace(
                0, self.max_disp-1, self.max_disp,
            )
            disp_sample = disp_sample.repeat(1,1,1,1,1).permute(0, 1, 4, 2, 3).contiguous()
            self.disp_regression = nn.Conv3d(1, 1, (self.max_disp, 1, 1), 1, 0, bias=False)
            self.disp_regression.weight.data = disp_sample
            self.disp_regression.weight.requires_grad = False

    def _init_data(self,left_imgs, right_imgs, **kwargs):
        '''
        description: 
        parameter: {*} Input tensor data (GT depth, pose, camera instrinsics etc. are in kwargs)
        return: {*}
        '''        
        left_imgs = left_imgs.reshape((left_imgs.shape[0],-1)+left_imgs.shape[1:] )
        right_imgs = right_imgs.reshape((right_imgs.shape[0],-1)+right_imgs.shape[1:] )
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

    def build_image_volume(self, leftImage, rightImage):
        device = leftImage.device
        B, C, H, W = leftImage.shape
        D = self.max_disp
        disp_sample = torch.linspace(0, D-1, D, device=device)
        disp_sample = disp_sample.reshape((1, D, 1, 1)).expand(B, D, H, W)
        # expand D dimension
        concat_leftImage = leftImage.unsqueeze(2).expand(B, C, D, H, W)
        concat_rightImage = rightImage.unsqueeze(2).expand(B, C, D, H, W)

        pad_val = torch.mean(torch.cat([leftImage,rightImage], dim=1).reshape(-1), dim=0)
        # shift target feature according to disparity samples
        concat_rightImage = inverse_warp_3d(concat_rightImage.float(), -disp_sample.float(), padding_mode='zeros')

        # mask out features in reference
        #concat_leftImage = concat_leftImage * (concat_rightImage > 0).type_as(leftImage) # fix the type bug when using half-float. ziming 21-7-8
        concat_rightImage[concat_rightImage==0] = pad_val # mean value to replace zero padding
        #concat_leftImage[concat_leftImage==0] = pad_val # mean value to replace zero padding
        #for i in range(192):
        #    vis_img_tensor(concat_rightImage[0,:,i,:,:], '/home/ziliu/vis/pixelnet/', f"concat_rightImage_{i}")
        #    vis_img_tensor(concat_leftImage[0,:,i,:,:], '/home/ziliu/vis/pixelnet/', f"concat_leftImage_{i}")
        # [B, 2C, D, H, W)
        image_volume = torch.cat((concat_leftImage, concat_rightImage), dim=1)

        return image_volume.detach()

    def build_image_volume_mat_nomask(self, leftImage, rightImage, max_disp=192, mask_left=False, 
                            img_pad_zeros=False, mask_template=None, indices=None):
        device = leftImage.device
        B, C, H, W = leftImage.shape
        D = self.max_disp

        if self.mask_template is None or (self.mask_template.shape[0],self.mask_template.shape[3],self.mask_template.shape[4])!=(leftImage.shape[0],leftImage.shape[2],leftImage.shape[3]):
            print("re-init mask template")
            #print("leftImage.shape: ", leftImage.shape)
            #print("mask_template.shape: ", self.mask_template.shape if self.mask_template is not None else "None")
            B, C, H, W = leftImage.shape
            self.mask_template = torch.ones(B, 2*3, max_disp, H, W, device=leftImage.device)
            for i in range(max_disp):
                self.mask_template[:, :, i, :, 0:i] = 0
      
        if self.indices is None or self.indices.shape != (D,W):
            #print("init indices")
            indices = torch.arange(W,device=device).view((1,W)).repeat((D,1))
            shifts = torch.arange(D,device=device).reshape(D,1)
            self.indices = (indices - shifts) % W
            print("indices: ", self.indices)

        rightImageseq = rightImage.unsqueeze(2).expand(B, C, D, H, W)
        rightImageseq = torch.gather(rightImageseq, 4, self.indices.reshape((1,1,D,1,W)).expand(B,C,D,H,W))
        leftImageseq = leftImage.unsqueeze(2).expand(B, C, D, H, W)

        image_volume = torch.cat([leftImageseq,rightImageseq], dim=1)
        #if mask_template is not None:
            ##print(f"mask_template {mask_template.shape}, image_volume {image_volume.shape}")
            #image_volume = image_volume * mask_template
        #    image_volume = image_volume.mul_(self.mask_template.detach())
            #image_volume = torch.where(mask_template>0, image_volume, torch.zeros_like(image_volume))
        return image_volume

    def build_image_volume_mat_halfmask(self, leftImage, rightImage, max_disp=192, mask_left=False, 
                            img_pad_zeros=False, mask_template=None, indices=None):
        device = leftImage.device
        B, C, H, W = leftImage.shape
        D = self.max_disp

        if self.mask_template is None or (self.mask_template.shape[0],self.mask_template.shape[3],self.mask_template.shape[4])!=(leftImage.shape[0],leftImage.shape[2],leftImage.shape[3]):
            print("re-init mask template")
            #print("leftImage.shape: ", leftImage.shape)
            #print("mask_template.shape: ", self.mask_template.shape if self.mask_template is not None else "None")
            B, C, H, W = leftImage.shape
            self.mask_template = torch.ones(B, C, max_disp, H, W, device=leftImage.device)
            for i in range(max_disp):
                self.mask_template[:, :, i, :, 0:i] = 0
      
        if self.indices is None or self.indices.shape != (D,W):
            #print("init indices")
            indices = torch.arange(W,device=device).view((1,W)).repeat((D,1))
            shifts = torch.arange(D,device=device).reshape(D,1)
            self.indices = (indices - shifts) % W

        rightImageseq = rightImage.unsqueeze(2).expand(B, C, D, H, W)
        rightImageseq = torch.gather(rightImageseq, 4, self.indices.reshape((1,1,D,1,W)).expand(B,C,D,H,W))
        leftImageseq = leftImage.unsqueeze(2).expand(B, C, D, H, W)

        rightImageseq = rightImageseq.mul_(self.mask_template.detach())

        image_volume = torch.cat([leftImageseq,rightImageseq], dim=1)

        return image_volume.detach()

    def build_image_volume_mat_mask(self, leftImage, rightImage, max_disp=192, mask_left=False, 
                            img_pad_zeros=False, mask_template=None, indices=None):
        device = leftImage.device
        B, C, H, W = leftImage.shape
        D = self.max_disp

        if self.mask_template is None or (self.mask_template.shape[0],self.mask_template.shape[3],self.mask_template.shape[4])!=(leftImage.shape[0],leftImage.shape[2],leftImage.shape[3]):
            print("re-init mask template")
            #print("leftImage.shape: ", leftImage.shape)
            #print("mask_template.shape: ", self.mask_template.shape if self.mask_template is not None else "None")
            B, C, H, W = leftImage.shape
            self.mask_template = torch.ones(B, 2*C, max_disp, H, W, device=leftImage.device)
            for i in range(max_disp):
                self.mask_template[:, :, i, :, 0:i] = 0
      
        if self.indices is None or self.indices.shape != (D,W):
            #print("init indices")
            indices = torch.arange(W,device=device).view((1,W)).repeat((D,1))
            shifts = torch.arange(D,device=device).reshape(D,1)
            self.indices = (indices - shifts) % W

        rightImageseq = rightImage.unsqueeze(2).expand(B, C, D, H, W)
        rightImageseq = torch.gather(rightImageseq, 4, self.indices.reshape((1,1,D,1,W)).expand(B,C,D,H,W))
        leftImageseq = leftImage.unsqueeze(2).expand(B, C, D, H, W)

        image_volume = torch.cat([leftImageseq,rightImageseq], dim=1)
        image_volume = image_volume.mul_(self.mask_template.detach())
        return image_volume.detach()

    def disp_predictor(self, final_costs):
        if not isinstance(final_costs, list) and not isinstance(final_costs, tuple):
            final_costs= [final_costs]
        pred_disps = []
        for i in range(len(final_costs)):
            cost_volume = final_costs[i]
            if cost_volume.dim() == 5:
                cost_volume = cost_volume.squeeze(1)
            if cost_volume.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)'
                                .format(cost_volume.dim()))

            # scale cost volume with alpha
            cost_volume = cost_volume * self.alpha

            if self.normalize:
                prob_volume = F.softmax(cost_volume, dim=1)
            else:
                prob_volume = cost_volume

            B, D, H, W = cost_volume.shape
            if self.use_conv_prediction:
                disp_map = self.disp_regression(prob_volume.unsqueeze(1)).squeeze(1)
                pred_disps.append(disp_map)
            else:
                disp_sample_pred_layer = torch.linspace(0, self.max_disp-1, self.max_disp, device=cost_volume.device)
                disp_sample = disp_sample_pred_layer.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
                # compute disparity: (BatchSize, 1, Height, Width)
                disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)
                pred_disps.append(disp_map)

        return pred_disps


    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        self._init_data(left_imgs, right_imgs, **kwargs)
        B, C, H, W = left_imgs.shape
        losses = dict()
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        # build image volume 
        
        if self.typeIV=="nomask":
            image_volume = self.build_image_volume_mat_nomask(leftImage, rightImage)
        elif self.typeIV=="halfmask":
            image_volume = self.build_image_volume_mat_halfmask(leftImage, rightImage)
        elif self.typeIV=="mask":
            image_volume = self.build_image_volume_mat_mask(leftImage, rightImage)
        else:
            raise ValueError("typeIV must be nomask halfmask or mask")
        
        # sample 
        image_volume_sampled = F.interpolate(image_volume,scale_factor=(1/self.sample_rate,1,1), mode='nearest')
        feature = self.backbone(image_volume_sampled)
        cost = self.neck(feature) # B 256 D/sample_rate H/4 W/4 
        cost = self.classify(cost[0]) # 256 -> 1 channel
        #print(cost.shape)
        #cost_full = F.interpolate(cost, scale_factor=(self.sample_rate,4,4), mode='trilinear')
        cost_full = F.interpolate(cost, size=(self.max_disp, H, W), mode='trilinear')
        disp = self.disp_predictor([cost_full])[0]

        if self.subnetwork:
            return disp

        #print(disp.shape)
        results = {}
        results["disps"] = disp
        results["right_disps"] = None
        if "left_gt_depth" in self.data[0].keys():
            gt_depth = torch.stack([self.data[ti]["left_gt_depth"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            results_depths = []
            for i in range(len(results["disps"])):
                results_depths.append((self.focal*self.baseline).reshape((-1,1,1,1)).repeat(1,1,results["disps"][i].shape[2],results["disps"][i].shape[3]) / (results["disps"][i]+1e-3))
            loss_left_gt_disp = self.loss(results_depths, gt_depth)
            losses.update(loss_left_gt_disp)
            return losses
        if "left_gt_disp" in self.data[0].keys():
            gt_disp = torch.stack([self.data[ti]["left_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #if torch.max(gt_disp)>self._MAX_DISP:
            #        self._MAX_DISP =torch.max(gt_disp)
            #        print(self._MAX_DISP)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_left_gt_disp = self.loss(results["disps"], gt_disp)
            losses.update(loss_left_gt_disp)
        """ 
        if "right_gt_disp" in self.data[0].keys() and results["right_disps"] is not None:
            gt_disp = torch.stack([self.data[ti]["right_gt_disp"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 1, self.height, self.width)
            #left_imgs = torch.stack([self.data[ti]["left_img"] for ti in range(self.num_frames)], 1).reshape(self.bs*self.num_frames, 3, self.height, self.width)
            loss_right_gt_disp = self.loss(results["right_disps"], gt_disp)
            losses.update(loss_right_gt_disp)
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
        end_time = time.time()
        out = self._do_test( left_imgs, right_imgs,**kwargs)
        print("time for one frame: ", time.time()-end_time)
        return out
        
    def _do_test(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        self._init_data(left_imgs, right_imgs, **kwargs)
        #from mmcv.runner import get_dist_info, init_dist, load_checkpoint
        #rank, _ = get_dist_info()
        B, C, H, W = left_imgs.shape
        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir
        network_pred_right_disparities = None
        #print("estimate depth in test stage...")
        leftImage =  left_imgs.reshape((-1, self.channel,self.height,self.width))
        rightImage = right_imgs.reshape((-1, self.channel,self.height,self.width))
        results ={}
        # build image volume 
        #t0=time.time()
        #with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        # build image volume 
        
        if self.typeIV=="nomask":
            image_volume = self.build_image_volume_mat_nomask(leftImage, rightImage)
        elif self.typeIV=="halfmask":
            image_volume = self.build_image_volume_mat_halfmask(leftImage, rightImage)
        elif self.typeIV=="mask":
            image_volume = self.build_image_volume_mat_mask(leftImage, rightImage)
        else:
            raise ValueError("typeIV must be nomask halfmask or mask")
        """
        image_volume = self.build_image_volume(leftImage, rightImage)
        """ 
        #torch.cuda.synchronize()
        # sample 
        #t1 = time.time()
        image_volume_sampled = F.interpolate(image_volume,scale_factor=(1/self.sample_rate,1,1), mode='nearest')
        feature = self.backbone(image_volume_sampled)
        #torch.cuda.synchronize()
        #t_feature = time.time()
        #print("feature time: ", t_feature-t1)
        cost = self.neck(feature) # B 256 D/sample_rate H/4 W/4
        #torch.cuda.synchronize() 
        #t_neck = time.time()
        #print("neck time: ", t_neck-t_feature)
        cost = self.classify(cost[0]) # 256 -> 1 channel
        #torch.cuda.synchronize()
        #t_classify = time.time()
        #print("classify time: ", t_classify-t_neck)
        if self.use_fast_prediction: 
            cost_full = F.interpolate(cost, size=(self.max_disp, cost.shape[3], cost.shape[4]), mode='trilinear')   
        else:
            cost_full = F.interpolate(cost, size=(self.max_disp, H, W), mode='trilinear')
        #torch.cuda.synchronize()
        #t_full = time.time()
        #print("full time: ", t_full-t_classify)
        disp = self.disp_predictor([cost_full])[0]
        if self.use_fast_prediction: disp = F.interpolate(disp*(W/disp.shape[3]), size=(H, W), mode='bilinear', align_corners=False)
        """ 
        torch.cuda.synchronize()
        #print(prof )
        t2 = time.time()
        print("pred conv time: ", t2-t_full)
        print("predict time: ", t2-t_neck)

        print(f"build image volume {t1-t0} network {t2-t1}, total {t2-t0}, FPS {1/(t2-t0)} ")
        if self.timer["f0_time"] == 0:
            self.timer["f0_time"] = t2 - t0
        else:
            self.timer["sum_time"] += t2 - t0
            self.timer["count"] += 1
            self.timer["avg_time"] = self.timer["sum_time"] / self.timer["count"]
            self.timer["fps"] = 1 / self.timer["avg_time"]
            print("avg_time: ", self.timer["avg_time"])
            print("fps: ", self.timer["fps"])
        """
        results["disps"] = disp
        results["right_disps"] = None
        pred_disparities = results["disps"] # keep the last prediction
        network_pred_right_disparities = results["right_disps"] if results["right_disps"] is not None else None
        #pred_disparities = [pred_disparities.reshape(pred_disparities.shape[0], 1, self.height, self.width)]
        #
        
        pred_disparities = [disp]
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
                    #left_pred_depth = (pred_W/self.width) * torch.stack([self.disp2depth_factor[bi]/(1e-10+pred_disparities[j].reshape((self.bs,self.num_frames,1,pred_H,pred_W))[bi,ti,:,:,:]) for bi in range(self.bs)])
                    left_pred_depth = (pred_W/self.width) * (kwargs["baseline"]* kwargs["focal"]).reshape(B,1,1,1) / (1e-3 + pred_disparities[j])
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
                            self.data[0]["right_pred_depth"][0].squeeze(1) ], 1).float().detach().cpu().numpy()
            else:
                all_pred_depthes = self.data[0]["left_pred_depth"][0].squeeze(1).unsqueeze(1).float().detach().cpu().numpy()
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
        


    def loss(self, pred, gt, **kwargs):
        losses = {}
        losses.update(self.disp_loss_func(pred, gt))
        return losses
    

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

    def forward(self, left_imgs, right_imgs =None, focal=None, baseline=None, intrinsics=None, return_loss=True, **kwargs):
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
        if focal is not None: kwargs['focal'] = focal
        if baseline is not None: kwargs['baseline'] = baseline
        if intrinsics is not None: kwargs['intrinsics'] = intrinsics
        return self.forward_test(left_imgs, right_imgs,  **kwargs)

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

    
if __name__ == '__main__':
    
    left_imgs = torch.randn(1, 3, 544, 960)
    right_imgs = torch.randn(1, 3, 544, 960)
    max_disp = 192
    model = PixelNet(sample_rate=8, 
        max_disp=max_disp,
        out_channels=16,
        backbone=dict(
            type='SqueezeNet3d',
            in_channels=6,
             
             ),
        neck=dict(type="FPN3d",
            in_channels=[ 128, 256,  384, 512 ], #[24, 116, 232, 464,],
            out_channels=16,#256,
            num_outs=4,
            out_indices=(0, ),
            up_sacles=(2,2,2), 
            start_level=0,
            end_level=-1,
            add_extra_convs=False,
            extra_convs_on_inputs=False,
            relu_before_extra_convs=False,
            no_norm_on_lateral=False,
            conv_cfg=dict(type='Conv3d'),
            #norm_cfg=dict(type='BN2d', requires_grad=True),
            #act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='trilinear'),),
        losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=(1.0, 0.7, 0.5),
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
            ),)
    model.init_weights()

    end_time = time.time()
    time = 0 
    for i in range(1000):
        outputs = model(left_imgs, right_imgs, return_loss=False)
        time += time.time() - end_time
        end_time = time.time()

    print("FPS: ", 1/time)


    