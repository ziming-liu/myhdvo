_base_ = [
    '../_base_/datasets/kittidepth_odometry_rgb_crop.py', 
]

max_disp = 192
model = dict(
    type="StereoHDVOPosesup",
    use_sup_pose=True,
    bidirection=True,
    depth_net=dict(
        type='CoEx',
        subnetwork=True,
        pretrain_official="pretrained/sceneflow.ckpt",
        cfg=dict(
            max_disparity= 192,
            backbone=dict(
                type= 'mobilenetv2_100',
                from_scratch= False,
                channels=dict(
                    mobilenetv3_large_100=[16,24,40,112,160],
                    mobilenetv2_120d=[24,32,40,112,192],
                    mobilenetv2_100=[16,24,32,96,160],
                    mnasnet_100=[16,24,40,96,192],
                    efficientnet_b0=[16,24,40,112,192],
                    efficientnet_b3a=[24,32,48,136,232],
                    mixnet_xl=[40,48,64,192,320],
                    dla34=[32,64,128,256,512]),
                layers=dict(
                    mobilenetv3_large_100=[1,2,3,5,6],
                    mobilenetv2_120d=[1,2,3,5,6],
                    mobilenetv2_100=[1,2,3,5,6],
                    mnasnet_100=[1,2,3,5,6],
                    efficientnet_b0=[1,2,3,5,6],
                    efficientnet_b3a=[1,2,3,5,6],
                    mixnet_xl=[1,2,3,5,6],
                    dla34=[1,2,3,5,6])
                
        ),
            corr_volume= True,
            gce= True,

            matching_head= 1,
            matching_weighted= False,

            spixel=dict(
                branch_channels= [32,48],
            ),
            aggregation=dict(
                disp_strides= 2,
                channels= [16,32,48],
                blocks_num= [2,2,2],
            ),
            regression=dict(
                top_k=2),
        ),
        losses=dict(
                type="DispL1Loss",
                start_disp=0,
                max_disp=max_disp,
                weight=1,
                weights=(1.0, 0.7, 0.5),
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
            ),
    ),
    pose_net=None,
     
    stereo_head=dict(
        type="StereoMatchingHead",
        photo_loss=dict(type='HUBERLoss', ratio=0.85),
        struct_loss=dict(type='SSIMLoss', ratio=0.15),
        grid_sample_type="pytorch",
        padding_mode="zeros"
    ),
    ddvo_head=dict(
        type="PoseDDVOHead",
        ddvo=dict(type="DirectVO_OpenRox", ifmask=1, disp_log=0, ifrobust=0),
        photo_loss=dict(type='HUBERLoss', ratio=0.85),
        struct_loss=dict(type='SSIMLoss', ratio=0.15),
        grid_sample_type="pytorch",
        padding_mode="zeros"
    ),
    smooth_loss=dict(type='DispSmoothLoss', ratio=0.1)
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ]
)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = True

optimizer = dict(type="Lion", lr=1e-4)
optimizer_config = dict(type="OptimizerHook", grad_clip=None)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=100
)

runner = dict(type='IterBasedRunner', max_iters=40000)
total_epochs = 40000
checkpoint_config = dict(by_epoch=False, interval=4000, save_optimizer=False)
evaluation = dict(interval=4000, metrics='EPE')
dist_params = dict(backend='nccl')

work_dir = 'work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss'