'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-10 00:57:00
LastEditors: Ziming Liu
LastEditTime: 2023-03-29 23:53:19

v8 use backbone f4f8f16f32, so the first level is /4 resolution not /2 as others
'''
import os.path as osp

_base_ = [
    '../_base_/datasets/scene_flow.py', '../_base_/iters_runtime.py',
    '../_base_/schedules/schedule_adamW_48k_lr5.py'
]

max_disp = 320
iters = 10
model=dict( 
        type="PyramidStereoSceneFlow2_1MonosLeft8Fixrange",
        in_channels=[256,256,256,256],
        mask_size=4,
        search_ranges=[8,8,8],
        iters=iters,
        max_disp=max_disp,
        backbone=dict(
            type="ConvNeXt",
            depths=[3, 3, 27, 3], 
            dims=[128, 256, 512, 1024],
            pretrained="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
            num_stages=4,
            out_indices=[0,1,2,3],
            stem_ratio=4,
        ),
        neck=dict(type="FPN",
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            num_outs=4,
            start_level=0,
            end_level=-1,
            add_extra_convs=False,
            extra_convs_on_inputs=False,
            relu_before_extra_convs=False,
            no_norm_on_lateral=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='bilinear'),),
        mono_head=dict(type="MonoDispHead", 
            max_depth=max_disp//32, 
            in_channel=256, 
            latent_channel=256, 
            out_channel=1,
            #losses=dict(
            #    type="SiLogLoss",
            #    name="mono",
            #    weights=[1,],
            #),
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=[0.8**(2*iters)],
                sparse=True,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="mono_l1_loss"
            ),
        ),
        disp_head=[dict(type="PyramidStereoHead2AdabinDelta", # use disp attne conv layer
            in_channels=[256*2], 
            adabins="pixel-wise",
            latent_dims=64,
            ############################################
            max_disp=max_disp,
            ############################################
            alpha=1., 
            normalize=True,
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=[  0.8 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=True,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        ),
        dict(type="PyramidStereoHead2AdabinDelta", # use disp attne conv layer
            in_channels=[256*2], 
            adabins="pixel-wise",
            latent_dims=64,
            ############################################
            max_disp=max_disp,
            ############################################
            alpha=1., 
            normalize=True,
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=[  0.8 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=True,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        ),
        dict(type="PyramidStereoHead2AdabinDelta", # use disp attne conv layer
            in_channels=[256*2], 
            adabins="pixel-wise",
            latent_dims=64,
            ############################################
            max_disp=max_disp,
            ############################################
            alpha=1., 
            normalize=True,
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=[  0.8 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        ) ]
)
 

work_dir = "work_dirs/pyramid_base_setting_cascade8oneSF200k10iters2"
 

dist_params = dict(backend='gloo')

find_unused_parameters = True