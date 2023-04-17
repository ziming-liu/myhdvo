'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 18:29:53
LastEditors: Ziming Liu
LastEditTime: 2023-03-21 23:40:01
'''


_base_ = [
    '../_base_/models/crestereo.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/cascade_stereo_16epoch.py'
]

max_disp = 192
model=dict( 
        type="CREStereo",
        max_disp=max_disp,
        mixed_precision=False, 
)

# optimizer
optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy="step", step=[10,12,14], gamma=1/2,)
# runtime settings
runner = dict(type='EpochBasedRunner' )
total_epochs=16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics='EPE', )



work_dir = "work_dirs/crestereo_sceneflow"

#resume_from = "work_dirs/crestereo_sceneflow/epoch_7.pth"
dist_params = dict(backend='gloo')

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        #dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
