'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2022-01-24 15:47:16
LastEditors: Ziming Liu
LastEditTime: 2023-03-29 18:27:40
'''

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
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
