'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 18:29:53
LastEditors: Ziming Liu
LastEditTime: 2023-04-18 02:58:55
'''


_base_ = [
    '../_base_/models/igevstereo.py',
    '../_base_/datasets/scene_flow_random320x768_igev.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamW_200k.py'
]
 


work_dir = "work_dirs/igevbase"

#resume_from = "work_dirs/crestereo_sceneflow/epoch_7.pth"
#dist_params = dict(backend='gloo')

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        #dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

find_unused_parameters=True
