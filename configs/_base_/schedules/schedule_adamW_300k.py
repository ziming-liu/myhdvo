'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-03-17 11:31:53
'''
# optimizer 
# adamW has faster training time and lower errors than adam
optimizer = dict(type='AdamW', lr=0.0001, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=4, loss_scale="dynamic",
                        grad_clip=None )

# learning policy, one cycle is faster to concerge with similar acc. 
lr_config = dict(policy='OneCycle',
    by_epoch=False, max_lr=0.0001, total_steps=300000,) 
# runtime settings
runner = dict(type='IterBasedRunner', max_iters= 300000)
total_epochs=300000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metrics='EPE', )

#fp16 = dict(loss_scale="dynamic")