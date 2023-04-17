'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-11 00:44:06
LastEditors: Ziming Liu
LastEditTime: 2023-03-15 17:14:17
'''
_base_ = [
    '../_base_/datasets/drivingstereo_random384x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_300k.py'
]
max_disp = 192

model=dict( 
        type="PyramidStereoSceneFlow2_1MonosLeft",
        in_channels=128,
        mask_size=2,
        search_ranges=[8,8,8],
        iters=10,
        max_disp=max_disp,
        backbone=dict(
            type="ResNet2",
            depth=18,
            pretrained='torchvision://resnet18',
            torchvision_pretrain=True,
            in_channels=3,
            num_stages=4,
            out_indices=(0,1,2,3 ),
            strides=(1,2,2,2  ),
            dilations=(1,1,1,1),
            style='pytorch',
            frozen_stages=-1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            norm_eval=False,
            partial_bn=False,
            with_cp=False
        ),
        neck=dict(type="FPN",
            in_channels=[64,128,256,512],
            out_channels=64,
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
            max_depth=max_disp//2, 
            in_channel=64, 
            latent_channel=64, 
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
                weights=(0.8 ** 20,),
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="mono_l1_loss"
            ),
        ),
        disp_head=dict(type="PyramidStereoHead2AdabinLight", # use disp attne conv layer
            in_channels=[128], 
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
                weights=[  0.8 ** (21 - i - 1) for i in range(1, 21)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        )
)
 
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

work_dir = 'work_dirs/pyramid_stereo_light_weightedloss_smallrange_kitti2015_384x512'
 
find_unused_parameters = True
