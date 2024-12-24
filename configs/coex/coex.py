'''
Developer=ACENTAURI team, INRIA institute
Author=Ziming Liu
Date=2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-09-10 16:53:55
'''
_base_ = [
    '../_base_/datasets/scene_flow_smallsize_b8.py', 
    '../_base_/schedules/schedule_adamW_48k_lion.py'
]

max_disp = 192
model=dict( 
        type='CoEx',
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
)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = True

optimizer = dict(type='Lion', lr=1e-4, )
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=256., grad_clip=dict(max_norm=35, norm_type=2) )


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=48000)
total_epochs=48000
checkpoint_config = dict(by_epoch=False, interval=12000)
evaluation = dict(interval=12000, metrics='EPE', )
dist_params = dict(backend='gloo')

work_dir = 'work_dirs/coex'
resume_from = None