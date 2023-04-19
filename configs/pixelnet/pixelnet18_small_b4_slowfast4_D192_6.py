'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-04-19 03:17:49
'''
_base_ = [
    '../_base_/datasets/scene_flow_smallsize_b8.py', 
    '../_base_/schedules/schedule_adamW_48k_lion.py'
]

max_disp = 192
model=dict( 
        type="PixelNetSlowFast4",
        sample_rate=1, 
        dense_channels=8, sparse_channels=64, max_disp=192,
        mask_left=False,
        fast_image_volume=True,
        backbone=dict(
           type='ResNet3dSlowFast',
            pretrained=None,
            resample_rate=32,  # tau
            speed_ratio=32,  # alpha
            channel_ratio=8,  # beta_inv
            out_indices=(0, 1, 2, 3),
            slow_pathway=dict(
                type='resnet3d',
                depth=18,
                in_channels=6,
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
                depth=18,
                in_channels=6,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=False)),
        neck_sparse=dict(type="FPN3d",
            in_channels=[256//4,512//4,1024//4,2048//4],
            out_channels=64,
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
            in_channels=[256//4//8,512//4//8,1024//4//8,2048//4//8],
            out_channels=8,
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

optimizer = dict(type='Lion', lr=1e-4, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=256., grad_clip=dict(max_norm=35, norm_type=2) )


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=48000)
total_epochs=48000
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metrics='EPE', )
dist_params = dict(backend='gloo')

work_dir = 'work_dirs/pixelnet18_small_b4_slowfast4_D192_6'
