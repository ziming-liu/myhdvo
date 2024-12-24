'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2024-02-08 14:49:39
'''
_base_ = [
    '../_base_/datasets/kittidepth_odometry_rgb_crop.py', 
]


max_disp = 192
model=dict(
    type="StereoHDVOPosesup",
    use_sup_pose=True,
    bidirection=True,
    depth_net=dict(
        type="PSMNet",
        subnetwork=True,
        backbone=dict(
            type="PSMNetSingle",
            in_planes=3,  # the in planes of feature extraction backbone
            with_cp=False
        ),
        disp_head=dict(type="PSMNetHead48onelevel",
            in_channels=[32*2], 
            ############################################
            disp_range=[0,max_disp//4,1], # [min, max, step]
            ############################################
            alpha=1., 
            normalize=True,
        ),
        smooth_loss = dict(type='DispSmoothLoss', ratio=0.1),
    ), # end of depth_net
    pose_net=None,
     
    stereo_head=dict(
        type="StereoMatchingHead",
         photo_loss=dict(type='L1Loss', ratio=0.85), #dict(type='L1Loss', ratio=0.85), # dict(type='ZNCCLoss', ratio=0.85), #
        struct_loss = dict(type='SSIMLoss', ratio=0.15),
        grid_sample_type="pytorch", 
        padding_mode="zeros",),
    ddvo_head=dict(
        type="PoseDDVOHead",
        ddvo=dict(type="DirectVO_OpenRox", ifmask=1, disp_log=0, ifrobust=0,),
         photo_loss=dict(type='L1Loss', ratio=0.85), #dict(type='L1Loss', ratio=0.85), # dict(type='ZNCCLoss', ratio=0.85), #
        struct_loss = dict(type='SSIMLoss', ratio=0.15),
        grid_sample_type="pytorch", 
        padding_mode="zeros",),
    #occ_mask=dict(
    #    type="STCMaskFast2",
    #    threshold_ratio=0.8,),
    #homo_mask=dict(
    #    type="HomoMaskFast",
    #    max_kernel=(5,5),
    #    avg_kernel=(5,5),
    #    threshold=2/255,
    #),
    smooth_loss = dict(type='DispSmoothLoss', ratio=0.1),


) # end of model

# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
         dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
log_level = 'INFO'
load_from = None
resume_from = None # 'work_dirs/hdvo_base_supposenet/iter_36000.pth'
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = True

optimizer = dict(type='DistShampoo', lr=1e-4, grafting_type="ADAM"
                            )

 
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
#optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=256., grad_clip=dict(max_norm=35, norm_type=2) )
#optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512., )
#optimizer_config = dict(type="OptimizerHook",  )
optimizer_config = dict(type="OptimizerHook", grad_clip=None )#grad_clip=dict(max_norm=35, norm_type=2) )


lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=100)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
total_epochs=40000
checkpoint_config = dict(by_epoch=False, interval=1000, save_optimizer=False)
evaluation = dict(interval=1000, metrics='EPE', )
dist_params = dict(backend='nccl')



work_dir = 'work_dirs/stereohdvo_posesup_s1_kittiodom'