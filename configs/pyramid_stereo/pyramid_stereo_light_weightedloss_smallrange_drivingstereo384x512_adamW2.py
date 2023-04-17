'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-11 00:44:06
LastEditors: Ziming Liu
LastEditTime: 2023-03-18 18:25:15
'''
_base_ = [
    '../_base_/datasets/drivingstereo_random384x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamW2_300k.py'
]
max_disp = 256
iters = 2
model=dict( 
        type="PyramidStereoSceneFlow2_1MonosLeftLight",
        in_channels=128,
        mask_size=2,
        search_ranges=[8,8,8],
        iters=iters,
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
        neck=None,
        mono_head=dict(type="MonoDispHead", 
            max_depth=max_disp//4, 
            in_channel=128, 
            latent_channel=128, 
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
                weights=(0.8 ** (2*iters),),
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
            latent_dims=128,
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
                weights=[  0.8 ** (2*iters+1 - i - 1) for i in range(1, 2*iters+1)],
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

work_dir = 'work_dirs/pyramid_stereo_light_weightedloss_smallrange_drivingstereo384x512_adamW2'

dist_params = dict(backend='nccl')


# optimizer 
# adamW has faster training time and lower errors than adam
optimizer = dict(type='AdamW', lr=0.0001, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=None )

# learning policy, one cycle is faster to concerge with similar acc. 
lr_config = dict(policy='OneCycle',
    by_epoch=False, max_lr=0.0001, total_steps=300000,) 
# runtime settings
runner = dict(type='IterBasedRunner', max_iters= 300000)
total_epochs=300000
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metrics='EPE', )

#fp16 = dict(loss_scale="dynamic")

find_unused_parameters = True
