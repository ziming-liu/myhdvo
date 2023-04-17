'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-03-13 18:23:21
'''
# optimizer
optimizer = dict(type='Adam', lr=1.5e-4, betas=(0.9, 0.999), )
optimizer_config = dict()
# learning policy
lr_config = dict(policy="cre", 
                 warmup='linear',
                warmup_iters=6000,
                warmup_ratio=1.0 / 3,
                warmup_by_epoch=False,
                const_range=0.6, min_lr=0.05)
# runtime settings
runner = dict(type='EpochBasedRunner' )
total_epochs= 600 
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics='EPE', )
