'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-06-14 16:11:53
'''
_base_ = [
    '../_base_/datasets/kittidepth_odometry_rgb_smallHW_bs2.py', 
]


max_disp = 192//2
model=dict( 
        type="IGEVStereo",
        use_unsup_loss=True,
        use_sup_loss=False,
        onlyleft=True,
        grad_hessian=False,
        cu_grad=False,
        backbone=dict(type="IGEVFeatureNet",
        ),
        args=dict(
                restore_ckpt=None,
                mixed_precision=False,
                train_datasets='sceneflow',
                valid_iters=3,
                train_iters=3,
                corr_implementation='reg',
                shared_backbone=True,
                corr_levels=2,
                corr_radius=4,
                n_downsample=2,
                slow_fast_gru=True,
                n_gru_layers=3,
                hidden_dims=[128]*3,
                max_disp=max_disp,
                val_init_disp=False,
                loss_gamma=0.9,
        ),

        photo_loss=dict(type='ZNCCLoss', ratio=0.85), #dict(type='L1Loss', ratio=0.85), # dict(type='ZNCCLoss', ratio=0.85), #
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


optimizer = dict(type='Adam', lr=0.0001)
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


work_dir = 'work_dirs/selfsupodometry_igev_camera23_smallHW_onlyleft_adam'
#load_from = 'work_dirs/pixelnet18_small_b8_slowonly/iter_48000.pth'
#resume_from = 'work_dirs/selfsupodometry_igev_camera23_smallHW_onlyleft_adam/iter_30000.pth'