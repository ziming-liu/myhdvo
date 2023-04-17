'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-04-06 12:59:51
'''
_base_ = [
    '../_base_/models/pixelnet.py',
    '../_base_/datasets/scene_flow_smallsize_b4.py', 
    '../_base_/schedules/schedule_adamW_48k_lion.py'
]

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
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=48000)
total_epochs=48000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metrics='EPE', )

work_dir = 'work_dirs/pixelnet_small_b8'
#resume_from = 'work_dirs/pixelnet_small_b8/iter_5000.pth'