'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-10 00:57:00
LastEditors: Ziming Liu
LastEditTime: 2023-03-30 19:30:14

v8 use backbone f4f8f16f32, so the first level is /4 resolution not /2 as others
'''
import os.path as osp

_base_ = [
    '../_base_/datasets/scene_flow_random320x768.py', '../_base_/iters_runtime.py',
   # '../_base_/schedules/schedule_adamW_48k.py'
]

max_disp = 192
iters = 2
model=dict( 
        type="PyramidStereoSceneFlow2_1MonosLeft11",
        in_channels=[64,64,64,64],
        mono_in_channels=2048,
        mask_size=4,
        search_ranges=[8,8,8],
        iters=iters,
        max_disp=max_disp,
        #output_branch="init",
        pretrained="https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.pth",
        backbone=dict(
                type='MobileOne',
                arch='s2',
                num_stages=4,
                out_indices=(0,1,2,3),
                stem_stride=2,
        ),
        neck=dict(type="FPN",
            in_channels=[96, 256, 640, 2048],
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
        mono_head=None,
        stereo_init_head = dict(type="PSMNetHead48",
            in_channels=[64*2], 
            ############################################
            disp_range=[0,max_disp//4,1], # [min, max, step]
            ############################################
            alpha=1./0.07, 
            normalize=True,
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
                name="init_loss"
            ),
        ),
        disp_head=[dict(type="PyramidStereoHead2AdabinDelta", # use disp attne conv layer
            in_channels=[64*2], 
            adabins="pixel-wise",
            if_adabins=False,
            latent_dims=32,
            ############################################
            max_disp=max_disp,
            ############################################
            alpha=1./0.07, 
            normalize=True,
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights= [ 0.9 ** i for i in range(iters-1, -1, -1)], # [  0.9 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        ),
        dict(type="PyramidStereoHead2AdabinDelta", # use disp attne conv layer
            in_channels=[64*2], 
            adabins="pixel-wise",
            if_adabins=False,
            latent_dims=32,
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
                weights=[  0.9 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        ),
        dict(type="PyramidStereoHead2AdabinDelta", # use disp attne conv layer
            in_channels=[64*2], 
            adabins="pixel-wise",
            if_adabins=False,
            latent_dims=32,
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
                weights= [ 0.9 ** i for i in range(iters-1, -1, -1)], # [  0.9 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        ) ]
)
 

work_dir = "work_dirs/pyramid_base_setting_cascade8convnext_SFx4size"
#resume_from = "work_dirs/pyramid_base_setting_cascade8convnext_SFx4size/iter_14000.pth"

 # yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

# adamW has faster training time and lower errors than adam
#optimizer = dict(type='Adam', lr=0.0001, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4,
#                        grad_clip=dict(max_norm=35, norm_type=2) )
# optimizer 
# adamW has faster training time and lower errors than adam
optimizer = dict(type='Lion', lr=1e-4, )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4,
#                        grad_clip=dict(max_norm=35, norm_type=2) )


# learning policy, one cycle is faster to concerge with similar acc. 
lr_config = dict(policy="step", step=[30000,36000,42000], gamma=1/2,)
                  #warmup="linear", warmup_iters=1000, warmup_ratio=1e-6,)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=48000)
total_epochs=48000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=500, metrics='EPE', )

dist_params = dict(backend='nccl')

find_unused_parameters = True