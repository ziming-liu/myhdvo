'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-04-02 11:59:50
'''
_base_ = [
    '../_base_/models/psmnet.py',
    '../_base_/datasets/scene_flow_smallsize_b8.py', 
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

work_dir = 'work_dirs/psmnet_small_b8'
#resume_from = 'work_dirs/psmnet_small_b8/iter_5000.pth'