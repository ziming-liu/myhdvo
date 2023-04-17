'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-03-10 00:04:05
'''
# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), )
optimizer_config = dict()
# learning policy
lr_config = dict(policy="Fixed")
# runtime settings
runner = dict(type='EpochBasedRunner' )
total_epochs=10
checkpoint_config = dict(interval=1)
evaluation = dict(interval=2, metrics='EPE', )
