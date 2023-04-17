'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-03-10 15:25:00
'''
_base_ = [
    '../_base_/models/psmnet.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/psmnet_10epoch.py'
]

# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), )
optimizer_config = dict()
# learning policy
lr_config = dict(policy="step", step=[10,12,14], gamma=1/2,)
# runtime settings
runner = dict(type='EpochBasedRunner' )
total_epochs=16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=2, metrics='EPE', )

load_from = None
resume_from = 'work_dirs/psmnet_random256x512/epoch_10.pth'

work_dir = 'work_dirs/psmnet_resume10_16_random256x512'