'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2024-02-02 23:08:07
'''
_base_ = [
    '../_base_/models/psmnet.py',
    '../_base_/datasets/scene_flow_smallsize_b8.py', 
]


# optimizer 
# adamW has faster training time and lower errors than adam
optimizer = dict(type='Adam', lr=5e-5, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=None )


# learning policy, one cycle is faster to concerge with similar acc. 
lr_config = dict(policy="step", step=[30000,36000,42000], gamma=1/2,)
                  #warmup="linear", warmup_iters=1000, warmup_ratio=1e-6,)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=48000)
total_epochs=48000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=500, metrics='EPE', )

#fp16 = dict(loss_scale=16.)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
         dict(type='TensorboardLoggerHook')
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

work_dir = 'work_dirs/psmnet_sup'
