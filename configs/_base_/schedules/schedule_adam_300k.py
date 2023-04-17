'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-03-16 15:14:24
'''
# optimizer
optimizer = dict(type='Adam', lr=0.0001, )
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', warmup='linear',
    warmup_iters=6000,
    warmup_ratio=1.0 / 3,
    by_epoch=False, gamma=0.5, step=[150000, 200000, 250000])
# runtime settings
runner = dict(type='IterBasedRunner', max_iters= 300000)
total_epochs=300000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=3000, metrics='EPE', )

#fp16 = dict(loss_scale="dynamic")