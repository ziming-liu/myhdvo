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
from ..builder import build_backbone, build_neck, build_disp_predictor,build_loss,build_head

from ..registry import STEREO_PREDICTOR

from ..losses import DispL1Loss
from .base_stereo import BaseStereo

from ...core.visulization import vis_depth_tensor,vis_img_tensor
from ..utils.inverse_warp_3d import inverse_warp_3d
import time 
from ..utils.temporal_warping2 import temporal_warp_c2r,  temporal_warp_r2c
from ..utils.stereo_warping2 import stereo_warp_r2l, stereo_warp_l2r
"""
this is a self-supervsised framework.
for now, we support two frames. 


"""

@STEREO_PREDICTOR.register_module()
class PixelNet2(nn.Module):
    """
    Base depth method. 

    """
    def __init__(self, backbone, losses, neck=None, stc_mask=None, lam_mask=None, alpha=1, normalize=True,
                  photo_loss=None, struct_loss=None, smooth_loss=None, 
                  sample_rate=4, max_disp=192, predict_format="depth", 
                  pretrained=None, tasks=["self_sup_eigen", ], internal_model=False,
                  learn_sparse_upsample=False, stereo_consistency=True,
                  graynormloss=False, graynorminput=False, **kwargs):
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
        super(PixelNet2, self).__init__()
        self.stereo_consistency = stereo_consistency # compute both left and right disparity
        self.graynorminput = graynorminput
        self.graynormloss = graynormloss
        self.internal_model = internal_model
        self.tasks = tasks
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

        if "self_sup_eigen" in tasks:
            assert photo_loss is not None and struct_loss is not None and smooth_loss is not None
            self.photo_loss_func = build_loss(photo_loss)
            self.struct_loss_func = build_loss(struct_loss)
            self.smooth_loss_func = build_loss(smooth_loss)
        
        self.classify = nn.Conv3d(256, 1, kernel_size=1, padding=0)
        self.learn_sparse_upsample = learn_sparse_upsample
        if learn_sparse_upsample:
            self.sparse_linear = nn.Linear(max_disp//sample_rate, max_disp)

        if stc_mask is not None:
            self.stc_mask = build_head(stc_mask)
        else:
            self.stc_mask = None
        if lam_mask is not None:
            self.lam_mask = build_head(lam_mask)
        else:
            self.lam_mask = None

    def build_image_volume_mat(self, left_imgs_BT, right_imgs_BT, max_disp=192, mask_left=False, 
                            img_pad_zeros=False, mask_template=None, indices=None):
        device = left_imgs_BT.device
        B, C, H, W = left_imgs_BT.shape
        max_disp = self.max_disp
        D = max_disp
        if self.mask_template is None or (self.mask_template.shape[0],self.mask_template.shape[-2],self.mask_template.shape[-1])!=(left_imgs_BT.shape[0],left_imgs_BT.shape[-2],left_imgs_BT.shape[-1]):
            print("re-init mask template")
            B, C, H, W = left_imgs_BT.shape
            self.mask_template = torch.ones(B, 2*C, max_disp, H, W, device=left_imgs_BT.device)
            for i in range(max_disp): self.mask_template[:, :, i, :, 0:i] = 0
        if self.indices is None or self.indices.shape != (D,W):
            indices = torch.arange(W,device=device).view((1,W)).repeat((D,1))
            shifts = torch.arange(D,device=device).reshape(D,1)
            self.indices = (indices - shifts) % W
        right_imgs_BTseq = right_imgs_BT.unsqueeze(2).expand(B, C, D, H, W)
        right_imgs_BTseq = torch.gather(right_imgs_BTseq, 4, self.indices.reshape((1,1,D,1,W)).expand(B,C,D,H,W))
        left_imgs_BTseq = left_imgs_BT.unsqueeze(2).expand(B, C, D, H, W)
        image_volume = torch.cat([left_imgs_BTseq,right_imgs_BTseq], dim=1)
        return image_volume.mul_(self.mask_template.detach())

    def disp_predictor(self, final_costs):
        if not isinstance(final_costs, list) and not isinstance(final_costs, tuple):
            final_costs= [final_costs]
        pred_disps = []
        for i in range(len(final_costs)):
            cost_volume = final_costs[i]
            if cost_volume.dim() == 5:
                cost_volume = cost_volume.squeeze(1)
            if cost_volume.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)' .format(cost_volume.dim()))
            cost_volume = cost_volume * self.alpha
            if self.normalize:
                prob_volume = F.softmax(cost_volume, dim=1)
            else:
                prob_volume = cost_volume
            B, D, H, W = cost_volume.shape
            disp_sample_pred_layer = torch.linspace(0, self.max_disp-1, self.max_disp, device=cost_volume.device)
            disp_sample = disp_sample_pred_layer.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)
            pred_disps.append(disp_map)
        return pred_disps

    def forward_train(self, left_imgs, right_imgs, **kwargs):
        """Defines the computation performed at every call when training."""
        assert len(left_imgs.shape) == 5
        B, T, C, H, W = left_imgs.shape
        left_imgs_BT, right_imgs_BT =  left_imgs.reshape((B*T, C, H, W)), right_imgs.reshape((B*T, C, H, W)) 
        # intrinsics +  extrinsics
        self.K_left = kwargs['K_left']
        self.K_right = kwargs['K_right'] if 'K_right' in kwargs else self.K_left
        self.focal_left, self.baseline = kwargs["focal_left"].reshape(B,1,1,1), kwargs["baseline"].reshape(B,1,1,1)
        self.focal_right = kwargs["focal_right"].reshape(B,1,1,1) if "focal_right" in kwargs else self.focal_left
        self.gt_poses = kwargs["pose"] # B T 4 4
        # 4D image volume  
        left_image_volume = self.build_image_volume_mat(left_imgs_BT, right_imgs_BT)
        right_image_volume = self.build_image_volume_mat(right_imgs_BT, left_imgs_BT)
        image_volume = torch.cat([left_image_volume, right_image_volume], dim=0) # B*T*2 2C D H W
        image_volume_sampled = F.interpolate(image_volume,scale_factor=(1/self.sample_rate,1,1), mode='nearest')
        # run model
        feature = self.backbone(image_volume_sampled.detach())
        cost = self.neck(feature) # B 256 D/sample_rate H/4 W/4 
        cost = self.classify(cost[0]) # 256 -> 1 channel
        if not self.learn_sparse_upsample:
            cost_full = F.interpolate(cost, scale_factor=(self.sample_rate,4,4), mode='trilinear')
        else:
            cost_full = self.sparse_linear( cost.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        raw_pred_disp = self.disp_predictor([cost_full])[0]
        # reshape disp 
        disp = raw_pred_disp.reshape(B,T,2,1,H,W)

        # compute losses
        losses = dict()
        
        left_disp_bt = disp[:,:,0,...].reshape((B*T,C,H,W))
        right_disp_bt = disp[:,:,1,...].reshape((B*T,C,H,W))
        smooth_loss_stereo = self.smooth_loss_func(left_disp_bt, left_imgs_BT.detach())
        smooth_loss_stereo = self.smooth_loss_func(right_disp_bt, right_imgs_BT.detach()) + smooth_loss_stereo
        losses.update({"loss_smooth_stereo": smooth_loss_stereo})
        # on left view  >> stereo loss
        stereo_warping_res_leftview = stereo_warp_r2l(right_imgs_BT.detach(), left_disp_bt ) # B*T C H W
        photo_loss_stereo_Lview = self.photo_loss_func(stereo_warping_res_leftview, left_imgs_BT)
        struct_loss_stereo_Lview = self.struct_loss_func(stereo_warping_res_leftview, left_imgs_BT)
         
        # on right view >> stereo loss
        stereo_warping_res_rightview = stereo_warp_l2r(left_imgs_BT.detach(), right_disp_bt )
        photo_loss_stereo_Rview = self.photo_loss_func(stereo_warping_res_rightview, right_imgs_BT) 
        struct_loss_stereo_Rview = self.struct_loss_func(stereo_warping_res_rightview, right_imgs_BT) 
         
        if self.lam_mask is not None:
            lam_mask_left = self.lam_mask(kwargs["raw_left_imgs"].reshape(B*T,C,H,W))
            lam_mask_right = self.lam_mask(kwargs["raw_right_imgs"].reshape(B*T,C,H,W))
            photo_loss_stereo_Lview *= lam_mask_left.detach()
            photo_loss_stereo_Rview *= lam_mask_right.detach()
            struct_loss_stereo_Lview *= lam_mask_left.detach()
            struct_loss_stereo_Rview *= lam_mask_right.detach()
            lam_mask_left = lam_mask_left.reshape(B,T,1,H,W).detach()
            lam_mask_right = lam_mask_right.reshape(B,T,1,H,W).detach()

        assert self.K_left.equal(self.K_right)
        assert self.focal_left.equal(self.focal_right) # have the same K focal, we can compute loss in a batch
        left_stc_masks = []
        right_stc_masks = []
        self.K_left = torch.cat([self.K_left, self.K_right], dim=0)
        self.focal_left = torch.cat([self.focal_left, self.focal_right], dim=0)
        self.baseline = self.baseline.repeat(2,1,1,1)
        for ref_id, cur_id in zip(range(0,T-1), range(1,T)):
            imgs_reference = torch.cat([left_imgs[:,ref_id,...],right_imgs[:,ref_id,...]], dim=0) #B*2 C H W
            imgs_target = torch.cat([left_imgs[:,cur_id,...],right_imgs[:,cur_id,...]], dim=0) #B*2 C H W
            disps_reference = torch.cat([disp[:,ref_id,0,...],disp[:,ref_id,1,...]], dim=0) #B*2 C H W
            disps_target = torch.cat([disp[:,cur_id,0,...],disp[:,cur_id,1,...]], dim=0) #B*2 C H W
            # stereo images on reference view >> temporal loss
            depth = (self.focal_left * self.baseline)  / (disps_reference+1e-8)
            cTr = torch.linalg.inv(self.gt_poses[:,cur_id,:,:]) @ self.gt_poses[:,ref_id,:,:] # inv(Tc) * Tr
            cTr = cTr.repeat(2,1,1) # left + right, use same camera poses
            temporal_warping_res_refview = temporal_warp_c2r(imgs_target.detach(), depth, cTr, 
                                            self.K_left, torch.linalg.inv(self.K_left),  ) # B*2 C H W
            photo_loss_temporal_refview = self.photo_loss_func(temporal_warping_res_refview, imgs_reference)
            struct_loss_temporal_refview = self.struct_loss_func(temporal_warping_res_refview, imgs_reference)
            # stereo image on target view >> temporal loss
            depth = (self.focal_left * self.baseline)  / (disps_target+1e-8)
            rTc = torch.linalg.inv(self.gt_poses[:,0,:,:]) @ self.gt_poses[:,1,:,:] # inv(Tr) * Tc
            rTc = rTc.repeat(2,1,1)
            temporal_warping_res_curview = temporal_warp_r2c(imgs_reference.detach(), depth, rTc,
                                            self.K_left, torch.linalg.inv(self.K_left),  )
            photo_loss_temporal_curview = self.photo_loss_func(temporal_warping_res_curview, imgs_target)  
            struct_loss_temporal_curview = self.struct_loss_func(temporal_warping_res_curview, imgs_target) 

            if self.lam_mask is not None:
                photo_loss_temporal_refview *= torch.cat([lam_mask_left[:,ref_id,...],lam_mask_right[:,ref_id,...]], dim=0).detach()
                photo_loss_temporal_curview *= torch.cat([lam_mask_left[:,cur_id,...],lam_mask_right[:,cur_id,...]], dim=0).detach()
            if self.stc_mask is not None:
                """
                   left right
                t0  00   01
                t1  10   11 
                """
                stc_mask_00 = self.stc_mask( stereo_warping_res_leftview.reshape(B,T,C,H,W)[:,ref_id], 
                                            temporal_warping_res_refview.reshape(B,2,C,H,W)[:,0],)
                stc_mask_01 = self.stc_mask( stereo_warping_res_rightview.reshape(B,T,C,H,W)[:,ref_id],
                                            temporal_warping_res_refview.reshape(B,2,C,H,W)[:,1],)
                stc_mask_10 = self.stc_mask( stereo_warping_res_leftview.reshape(B,T,C,H,W)[:,cur_id],
                                            temporal_warping_res_curview.reshape(B,2,C,H,W)[:,0],)
                stc_mask_11 = self.stc_mask( stereo_warping_res_rightview.reshape(B,T,C,H,W)[:,cur_id],
                                            temporal_warping_res_curview.reshape(B,2,C,H,W)[:,1],)
                photo_loss_temporal_refview *= torch.cat([stc_mask_00,stc_mask_01], dim=0).detach()
                photo_loss_temporal_curview *= torch.cat([stc_mask_10,stc_mask_11], dim=0).detach()
                struct_loss_temporal_refview *= torch.cat([stc_mask_00,stc_mask_01], dim=0).detach()
                struct_loss_temporal_curview *= torch.cat([stc_mask_10,stc_mask_11], dim=0).detach()
                # save stc masks for stereo losses
                if len(left_stc_masks) == 0:
                    left_stc_masks.append( stc_mask_00)
                left_stc_masks.append( stc_mask_10)
                if len(right_stc_masks) == 0:
                    right_stc_masks.append( stc_mask_01)
                right_stc_masks.append( stc_mask_11)

            losses.update({f"loss_photo_temporal_f{ref_id},f{cur_id}": photo_loss_temporal_refview+photo_loss_temporal_curview})
            losses.update({f"loss_struct_temporal_f{ref_id},f{cur_id}": struct_loss_temporal_refview+struct_loss_temporal_curview})
        ########### end for temporal loss ###########
        if self.stc_mask is not None:
            # stereo loss 
            photo_loss_stereo_Lview *= torch.cat(left_stc_masks, dim=0).detach()
            photo_loss_stereo_Rview *= torch.cat(right_stc_masks, dim=0).detach()
            struct_loss_stereo_Lview *= torch.cat(left_stc_masks, dim=0).detach()
            struct_loss_stereo_Rview *=  torch.cat(right_stc_masks, dim=0).detach()
        losses.update({"loss_photo_stereo": photo_loss_stereo_Lview+photo_loss_stereo_Rview})
        losses.update({"loss_struct_stereo": struct_loss_stereo_Lview+struct_loss_stereo_Rview})
        return losses


    def forward_test(self, left_imgs, right_imgs, **kwargs):
        return self._do_test( left_imgs, right_imgs,**kwargs)
    def _do_test(self, left_imgs, right_imgs,**kwargs):

        if len(left_imgs.shape) == 5: B, T, C, H, W = left_imgs.shape
        elif len(left_imgs.shape) == 4: B, C, H, W = left_imgs.shape
        else: raise ValueError("invalid input shape")
        if "self_sup_eigen" in self.tasks:
            assert T==2, "only support 2 frames"
            left_imgs_BT, right_imgs_BT =  left_imgs[:,0,:,:,:], right_imgs[:,0,:,:,:] # only compute the reference depth. 
            gray_images_left, gray_images_right = kwargs["gray_left_imgs"].reshape(B,T,1,H,W).float(), kwargs["gray_right_imgs"].reshape(B,T,1,H,W).float()
            normed_gray_images_left, normed_gray_images_right =  (2 * (gray_images_left / 255.0) - 1.0),  (2 * (gray_images_right / 255.0) - 1.0)
            gray_images_left, gray_images_right = normed_gray_images_left.float(), normed_gray_images_right.float()
        else:
            left_imgs_BT, right_imgs_BT =  left_imgs.reshape((-1, C, H, W)), right_imgs.reshape((-1, C, H, W))
        
        # K, pose ... 
        intrinsics = kwargs['intrinsics']
        self.K_left, self.K_right = intrinsics[:,0,:,:].type(torch.float32), intrinsics[:,1,:,:].type(torch.float32)
        self.focal_left, self.baseline = kwargs["focal_left"].reshape(B,1,1,1), kwargs["baseline"].reshape(B,1,1,1)
        self.disp2depth_factor = self.focal_left * self.baseline
        # labels if existing
        if "depth_sup" in self.tasks or "stereo_depth_sup" in self.tasks:
            self.gt_depths_left = kwargs['left_depths'] # B T 1 H W 
        if "stereo_depth_sup" in self.tasks: self.gt_depths_right = kwargs['right_depths'] # # B T 1 H W 
        if "disp_sup" in self.tasks or "stereo_disp_sup" in self.tasks:
            self.gt_disps_left = kwargs['left_disps']
        if "stereo_disp_sup" in self.tasks:  self.gt_disps_right = kwargs['right_disps']
        # 4D image volume is done here
        if self.graynorminput:
            image_volume = self.build_image_volume_mat(normed_gray_images_left[:,0,:,:,:], normed_gray_images_right[:,0,:,:,:] )
        else:
            image_volume = self.build_image_volume_mat(left_imgs_BT, right_imgs_BT)
        image_volume_sampled = F.interpolate(image_volume,scale_factor=(1/self.sample_rate,1,1), mode='nearest')

        # run model
        feature = self.backbone(image_volume_sampled)
        cost = self.neck(feature) # B 256 D/sample_rate H/4 W/4 
        cost = self.classify(cost[0]) # 256 -> 1 channel
        #print(cost.shape)
        if not self.learn_sparse_upsample:
            cost_full = F.interpolate(cost, scale_factor=(self.sample_rate,4,4), mode='trilinear')
        else:
            cost_full = self.sparse_linear( cost.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        disp = self.disp_predictor([cost_full])[0]


        outputs = [[],[],[],[],[],[]] # [pred_depths BxNxHxW], [gt_depths BxNxHxW], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir

        if "self_sup_eigen" in self.tasks:
            disp[disp==0] = 1e-6
            pred_depth = self.disp2depth_factor / disp
            if not self.internal_model: 
                pred_depth = pred_depth.reshape((B,1,H,W)).cpu().numpy()
            outputs[0] = pred_depth
            return outputs
        if "depth_sup" in self.tasks:
            disp[disp==0] = 1e-6
            pred_depth = self.disp2depth_factor / disp
            if not self.internal_model: 
                pred_depth = pred_depth.reshape((B,1,H,W)).cpu().numpy()
                gt_depth = self.gt_depths_left[:,0,:,:].reshape((B,1,H,W)).cpu().numpy()
            outputs[0] = pred_depth
            outputs[1] = gt_depth
            return outputs

        if "disp_sup" in self.tasks:
            if not self.internal_model: 
                pred_disp = disp.reshape((B,1,H,W)).cpu().numpy()
                gt_disp = self.gt_disps_left[:,0,:,:].reshape((B,1,H,W)).cpu().numpy()
            outputs[0] = pred_disp
            outputs[1] = gt_disp
            return outputs
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

    
