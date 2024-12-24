'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:49:28
LastEditors: Ziming Liu
LastEditTime: 2023-04-26 18:42:03
'''
# optimizer 
lr = 0.0004
num_steps = 200000
# adamW has faster training time and lower errors than adam
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-05, eps=1e-8 )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(grad_clip=dict(max_norm=1.0)) #dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=4, loss_scale="dynamic",
                   #     grad_clip=None )

# learning policy, one cycle is faster to concerge with similar acc. 
# https://mmcv.readthedocs.io/en/v1.5.1/_modules/mmcv/runner/hooks/lr_updater.html
# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
lr_config = dict(policy='OneCycle', max_lr=lr, total_steps=num_steps+100,
            pct_start=0.01,  anneal_strategy='linear',  # same as igev config cycle_momentum=False,
            by_epoch=False,
     ) 

# runtime settings
runner = dict(type='FreezeBNIterBasedRunner', max_iters= num_steps)
checkpoint_config = dict(by_epoch=False, interval=num_steps//40)
evaluation = dict(interval=num_steps//40, metrics='EPE', )
total_epochs = num_steps
#fp16 = dict(loss_scale="dynamic")