'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-04-06 12:59:42
'''
_base_ = [
    '../_base_/models/pixelnet.py',
    '../_base_/datasets/scene_flow_smallsize_b4.py', 
    '../_base_/schedules/schedule_adamW_48k_lion.py'
]

max_disp = 192
model=dict( 
        type="PixelNet",
        sample_rate=4, 
        max_disp=max_disp,
        backbone=dict(
            type='ResNet3dSlowOnly',
            in_channels=6,
            depth=18,
            pretrained='torchvision://resnet50',
            lateral=False,
            out_indices=(0,1,2,3, ),
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        neck=dict(type="FPN3d",
            in_channels=[256//4,512//4,1024//4,2048//4],
            out_channels=256,
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
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = False

optimizer = dict(type='Lion', lr=1e-4, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2) )


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=48000)
total_epochs=48000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=500, metrics='EPE', )

work_dir = 'work_dirs/pixelnet18_small_b4'
#resume_from = 'work_dirs/pixelnet18_small_b4/iter_100.pth'