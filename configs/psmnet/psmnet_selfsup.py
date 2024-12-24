'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2024-02-02 23:07:45
'''
_base_ = [
    '../_base_/datasets/kittidepth_odometry_rgb_smallHW_bs2.py', 
]


max_disp = 192//2
model=dict( 
        type="PSMNet",
        use_unsup_loss=True,
        use_sup_loss=False,
        onlyleft=True,
        grad_hessian=False,
        cu_grad=False,
        grid_sample_type="pytorch",
        backbone=dict(
            type="PSMNetSingle",
            in_planes=3,  # the in planes of feature extraction backbone
            with_cp=False
        ),
        disp_head=dict(type="PSMNetHead48",
            in_channels=[32*2], 
            ############################################
            disp_range=[0,max_disp//4,1], # [min, max, step]
            ############################################
            alpha=1., 
            normalize=True,
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=(1.0, 0.7, 0.5),
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
            ),
        ),

        photo_loss=dict(type='L1Loss', ratio=0.85), # dict(type='ZNCCLoss', ratio=0.85), #
        struct_loss = dict(type='SSIMLoss', ratio=0.15),
        smooth_loss = dict(type='DispSmoothLoss', ratio=0.1),
        #lam_mask =dict(type="HomoMaskFast",  num_views=1, num_frames=2, num_level=1, 
        #                kernel1_size=(3,3), kernel2_size=(5,5), threshold=2),
        #stc_mask = dict(type="STCMaskUseLossWarp", num_views=1, num_frames=2,  
        #                num_level=1, error_metric = [ #dict(type="L1Loss", ratio=1),
        #        dict(type="ZNCCLoss", kernel_size=(15,15), ratio=1)],  threshold_type="abs",  threshold_ratio = 1. ),
 
)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
         dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = True


optimizer = dict(type='Adam', lr=1e-4)
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
#optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=256., grad_clip=dict(max_norm=35, norm_type=2) )
#optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512., )
optimizer_config = dict(type="OptimizerHook",  )

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=100)

 

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
total_epochs=40000
checkpoint_config = dict(by_epoch=False, interval=5000, save_optimizer=False)
evaluation = dict(interval=2000, metrics='EPE', )
dist_params = dict(backend='gloo')


work_dir = 'work_dirs/psmnet_selfsup'
