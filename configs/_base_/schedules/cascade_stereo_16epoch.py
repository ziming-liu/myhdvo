'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-03-22 22:55:52
'''
# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), )
optimizer_config = dict()
# learning policy
lr_config = dict(policy="step", step=[10,12,14], gamma=1/2,)
# runtime settings
runner = dict(type='EpochBasedRunner' )
total_epochs=16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics='EPE', )
