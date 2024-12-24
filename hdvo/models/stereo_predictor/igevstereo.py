'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-04-18 00:32:00
LastEditors: Ziming Liu
LastEditTime: 2024-01-19 17:03:38
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.igev_feature_net import *
from .igevstereo_submodules.update import BasicMultiUpdateBlock
from .igevstereo_submodules.geometry import Combined_Geo_Encoding_Volume
from .igevstereo_submodules.submodule import *
import time
from .base_stereo import BaseStereo
from ..registry import STEREO_PREDICTOR
from ..builder import build_backbone,build_loss,build_head
from ..utils.temporal_warping import temporal_warp_c2r,  temporal_warp_r2c, temporal_warp_core
from ..utils.stereo_warping import stereo_warp_r2l, stereo_warp_l2r


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

@STEREO_PREDICTOR.register_module()
class IGEVStereo(BaseStereo):
    def __init__(self, backbone, args, test_iters=32, 
                 photo_loss=None, struct_loss=None, smooth_loss=None, 
                 grad_hessian=False, cu_grad_noesm=False, cu_grad=False,
                  use_sup_loss=True, use_unsup_loss=False, onlyleft=True,
                   lam_mask=None, stc_mask=None, **kwargs):
        super().__init__(backbone, **kwargs)
        self.cu_grad_noesm = cu_grad_noesm
        self.cu_grad = cu_grad
        self.grad_hessian = grad_hessian
        self.onlyleft = onlyleft
        self.lam_mask = lam_mask
        self.stc_mask = stc_mask

        self.use_sup_loss = use_sup_loss
        self.use_unsup_loss = use_unsup_loss
        assert use_sup_loss != use_unsup_loss, "sup_loss and unsup_loss cannot be both True or False"

        self.photo_loss_func = build_loss(photo_loss) if photo_loss is not None else None
        self.struct_loss_func = build_loss(struct_loss) if struct_loss is not None else None
        self.smooth_loss_func = build_loss(smooth_loss) if smooth_loss is not None else None

        self.args = args
        self.test_iters = test_iters

        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = build_backbone(backbone)

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        self.freeze_bn()

        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)
        

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp
    def forward_train_subnetwork(self, left_imgs, right_imgs, iters=12, flow_init=None, test_mode=False, **kwargs):
        iters = self.args.train_iters
        #self.freeze_bn()
        #for k, v in self.named_parameters():   
        #    print("k:{} v:{} ".format(k, v.requires_grad))
        if len(left_imgs.shape)>4:
            B, T, C, H, W = left_imgs.shape
            image1 = left_imgs.reshape((B*T, C, H, W))
            image2 = right_imgs.reshape((B*T, C, H, W))
        elif len(left_imgs.shape)==4:
            B, C,H, W = left_imgs.shape
            image1 = left_imgs
            image2 = right_imgs
        
        if 'intrinsics' in kwargs:
            self.K = kwargs['intrinsics'].reshape((B*2,)+kwargs['intrinsics'].shape[-2:]).float()
        if 'focal' in kwargs:
            self.focal_left, self.baseline = kwargs['focal'].reshape(B,1,1,1), kwargs["baseline"].reshape(B,1,1,1)
            self.focal_right =  self.focal_left
        if 'pose' in kwargs:
            self.gt_poses = kwargs["pose"] # B T 4 4
        
        
        """ Estimate disparity between pair of frames """
        #image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        #image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4)
            
            del prob, gwc_volume

            #if self.args.val_init_disp: # learn upsample weights
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]


        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(self.args.train_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if test_mode and itr < self.args.train_iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)

        disp_preds = [init_disp] + disp_preds
         
        return disp_preds


    def forward_train(self, left_imgs, right_imgs, iters=12, flow_init=None, test_mode=False, **kwargs):
        iters = self.args.train_iters
        #self.freeze_bn()
        #for k, v in self.named_parameters():   
        #    print("k:{} v:{} ".format(k, v.requires_grad))
        if len(left_imgs.shape)>4:
            B, T, C, H, W = left_imgs.shape
            image1 = left_imgs.reshape((B*T, C, H, W))
            image2 = right_imgs.reshape((B*T, C, H, W))
        elif len(left_imgs.shape)==4:
            B, C,H, W = left_imgs.shape
            image1 = left_imgs
            image2 = right_imgs
        
        if 'intrinsics' in kwargs:
            self.K = kwargs['intrinsics'].reshape((B*2,)+kwargs['intrinsics'].shape[-2:]).float()
        if 'focal' in kwargs:
            self.focal_left, self.baseline = kwargs['focal'].reshape(B,1,1,1), kwargs["baseline"].reshape(B,1,1,1)
            self.focal_right =  self.focal_left
        if 'pose' in kwargs:
            self.gt_poses = kwargs["pose"] # B T 4 4
        
        
        """ Estimate disparity between pair of frames """
        #image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        #image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4)
            
            del prob, gwc_volume

            #if self.args.val_init_disp: # learn upsample weights
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]


        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(self.args.train_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if test_mode and itr < self.args.train_iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        
        loss = {}
        if self.use_sup_loss:
            loss["loss"], _ = self.sequence_loss(disp_preds, init_disp, kwargs["left_disps"], None, loss_gamma=self.args.loss_gamma, max_disp=self.args.max_disp)
        if self.use_unsup_loss:
            for i in range(len(disp_preds)):
                loss.update(self.unsup_loss(disp_preds[i], left_imgs, right_imgs, None, None, level=i))
        return loss 

    def sequence_loss(self, disp_preds, disp_init_pred, disp_gt, valid=None, loss_gamma=0.9, max_disp=192):
        """ Loss function defined over sequence of flow predictions """
        if valid is None:
            valid = disp_gt < 512
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        disp_loss = 0.0
        mag = torch.sum(disp_gt**2, dim=1, keepdim=True).sqrt()
        valid = ((valid >= 0.5) & (mag < max_disp))#.unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()


        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
            disp_loss += i_weight * i_loss[valid.bool()].mean()

        epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return disp_loss, metrics

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
        stereo_warping_res_leftview = stereo_warp_r2l(right_imgs_BT, left_disp_bt, padding_mode="border", grad_hessian=self.grad_hessian, cu_grad_noesm=self.cu_grad_noesm, cu_grad=self.cu_grad, gt_map=left_imgs_BT ) # B*T C H W
        unvalid_mask_leftview = (stereo_warping_res_leftview != 0).all(dim=1, keepdim=True).long()
        photo_loss_stereo_Lview = self.photo_loss_func(stereo_warping_res_leftview, left_imgs_BT)
        struct_loss_stereo_Lview = self.struct_loss_func(stereo_warping_res_leftview, left_imgs_BT)
        #vis_depth_tensor(stereo_warping_res_leftview, "/home/ziliu/vis/", "leftwarping")
        #vis_depth_tensor(left_imgs_BT, "/home/ziliu/vis/",  "leftgt")
        
        # on right view >> stereo loss
        if not self.onlyleft:
            stereo_warping_res_rightview = stereo_warp_l2r(left_imgs_BT, right_disp_bt,  padding_mode="border", grad_hessian=self.grad_hessian, cu_grad_noesm=self.cu_grad_noesm, cu_grad=self.cu_grad, gt_map=right_imgs_BT ) # B*T C H W  )
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
                                            self.K_left, torch.linalg.inv(self.K_left.cpu()).cuda(),  padding_mode="border", grad_hessian=self.grad_hessian, cu_grad_noesm=self.cu_grad_noesm, cu_grad=self.cu_grad, gt_map=imgs_reference  ) # B*2 C H W
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
                                            self.K_left, torch.linalg.inv(self.K_left.cpu()).cuda(), padding_mode="border", grad_hessian=self.grad_hessian, cu_grad_noesm=self.cu_grad_noesm, cu_grad=self.cu_grad, gt_map=imgs_target    )
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


    def forward_test(self, left_imgs, right_imgs, **kwargs):
        if len(left_imgs.shape)>4:
            B, T, C, H, W = left_imgs.shape
            image1 = left_imgs[:,0]
            image2 = right_imgs[:,0]
        elif len(left_imgs.shape)==4:
            B, C,H, W = left_imgs.shape
            image1 = left_imgs
            image2 = right_imgs
            

        t0 = time.time()
        #with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        #image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        #image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        #with autocast(enabled=self.args.mixed_precision):
        features_left = self.feature(image1)
        features_right = self.feature(image2)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.max_disp//4)
        
        #del prob, gwc_volume

        #if self.args.val_init_disp: # learn upsample weights
        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]


        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(self.args.valid_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if itr < self.args.valid_iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        torch.cuda.synchronize()

        
        outputs = [[],[],[],[],[],[]] # [pred_depths], [gt_depths], [pred_masks/gtocclumask],[remove gt mask], [pred_poses],[gt_poses] frame_dir

        if self.args.val_init_disp:
            outputs[0] = init_disp.detach().cpu().numpy().reshape(B, 1, H, W)
        else:
            outputs[0] = disp_up.detach().cpu().numpy().reshape(B, 1, H, W)
        
        if 'left_disps' in kwargs:
            outputs[1] = kwargs['left_disps'].cpu().numpy().reshape(B, 1, H, W)
        
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
        #print(prof)
        return outputs
