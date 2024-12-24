'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-09-13 19:04:46
'''
_base_ = [
    #'../_base_/models/pixelnet.py',
    '../_base_/datasets/scene_flow_smallsize_b8.py', 
   # '../_base_/schedules/schedule_adamW_48k_lion.py'
]

max_disp = 192
model=dict( 
        type="PixelNet",
        sample_rate=2, 
        max_disp=max_disp,
        out_channels=8,
        backbone=dict(
            type='MobileNetV2_3d',
            in_channels=6,
            width_mult=1/4,
            last_channels=128,
             ),
        neck=dict(type="FPN3d",
            in_channels=[int(n*1/4) for n in [16, 24, 32, 96, ]],
            out_channels=8,#256,
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
            ),
)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = False

optimizer = dict(type='Lion', lr=1e-3, )
#optimizer = dict(type='DistShampoo', lr=1e-3,
#                        grafting_type='AdamW'    )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2) )

# learning policy, one cycle is faster to concerge with similar acc. 
#lr_config = dict(policy="step", step=[30000,36000,42000,50000,58000,64000,72000,80000,88000,96000,104000,184000,192000,200000], gamma=1/2,)
#                  #warmup="linear", warmup_iters=1000, warmup_ratio=1e-6,)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=1000)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=400000)
total_epochs=400000
checkpoint_config = dict(by_epoch=False, interval=8000, save_optimizer=False)
evaluation = dict(interval=8000, metrics='EPE', )

work_dir = 'work_dirs/mobilenetv2_3d_small_b4_slowfast4_d192_24_s3'
load_from = 'work_dirs/mobilenetv2_3d_small_b4_slowfast4_d192_24_s2/iter_144000.pth'
#resume_from = 'work_dirs/mobilenetv2_3d_small_b4_slowfast4_d192_24_s3/iter_144000.pth'