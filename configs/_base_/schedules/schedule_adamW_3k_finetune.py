'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-03-27 17:06:28
'''
# optimizer 
# optimizer 
# adamW has faster training time and lower errors than adam
optimizer = dict(type='Adam', lr=0.0002, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=None )

# learning policy, one cycle is faster to concerge with similar acc. 
lr_config = dict(policy="step", step=[2000,2500], gamma=1/2,
                  warmup="linear", warmup_iters=1000, warmup_ratio=1e-6,)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=3000)
total_epochs=3000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metrics='EPE', )