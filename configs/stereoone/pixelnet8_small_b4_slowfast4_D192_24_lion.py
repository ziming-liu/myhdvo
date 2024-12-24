'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2024-02-07 15:57:26
'''
_base_ = [
    '../_base_/datasets/scene_flow_smallsize_b8.py', 
   # '../_base_/schedules/schedule_adamW_48k_lion.py'
]

max_disp = 192
model=dict( 
        type="PixelNetSlowFast4",
        sample_rate=4, 
        dense_channels=4, sparse_channels=16, max_disp=max_disp,
        mask_left=False,
        fast_image_volume=True,
        backbone=dict(
           type='ResNet3dSlowFast',
            pretrained=None,
            resample_rate=8,  # tau
            speed_ratio=4,  # alpha
            channel_ratio=4,  # beta_inv
            out_indices=(0, 1, 2, 3),
            slow_pathway=dict(
                type='resnet3d',
                depth=8,
                in_channels=6,
                base_channels=16,
                pretrained=None,
                lateral=True,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=False),
            fast_pathway=dict(
                type='resnet3d',
                depth=8,
                in_channels=6,
                pretrained=None,
                lateral=False,
                base_channels=4,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=False)),
        neck_sparse=dict(type="FPN3d",
            in_channels=[256//16,512//16,1024//16,2048//16], # slow path channel / 16
            out_channels=16,
            num_outs=4,
            out_indices=(0, ),
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
        neck_dense=dict(type="FPN3d",
            in_channels=[256//8//8,512//8//8,1024//8//8,2048//8//8],
            out_channels=4,
            num_outs=4,
            out_indices=(0, ),
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
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = False

optimizer = dict(type='Lion', lr=1e-3, )
#optimizer = dict(type='DistShampoo', lr=1e-2,
#                        grafting_type='AdamW'    )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2) )

# learning policy, one cycle is faster to concerge with similar acc. 
#lr_config = dict(policy="step", step=[30000,36000,42000,], gamma=1/2,)
                  #warmup="linear", warmup_iters=1000, warmup_ratio=1e-6,)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=1000)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=200000)
total_epochs=200000
checkpoint_config = dict(by_epoch=False, interval=8000, save_optimizer=False)
evaluation = dict(interval=8000, metrics='EPE', )

dist_params = dict(backend='nccl')

work_dir = 'work_dirs/pixelnet8_small_b4_slowfast4_D192_24_lion'
