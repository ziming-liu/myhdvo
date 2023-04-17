'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-11 00:44:06
LastEditors: Ziming Liu
LastEditTime: 2023-03-23 10:56:21
'''
_base_ = [
    '../_base_/models/pyramid_stereo2_1monos.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/cascade_stereo_16epoch.py'
]
max_disp = 192

model=dict( 
        type="PyramidStereoSceneFlow2_1Monos",
        in_channels=192,
        mask_size=4,
        iters=2,
        backbone=dict(
            type="ConvNeXt",
            depths=[3, 3, 27, 3], 
            dims=[96, 192, 384, 768],
            pretrained="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
            num_stages=4,
            out_indices=[0,1,2,3],
            stem_ratio=2,
        ),
        neck=dict(type="FPN",
            in_channels=[96, 192, 384, 768],
            out_channels=192,
            num_outs=4,
            start_level=0,
            end_level=-1,
            add_extra_convs=False,
            extra_convs_on_inputs=False,
            relu_before_extra_convs=False,
            no_norm_on_lateral=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='bilinear'),),
        mono_head=dict(type="MonoDispHead", 
            max_depth=max_disp//2, 
            in_channel=192, 
            latent_channel=192, 
            out_channel=1,
            #losses=dict(
            #    type="SiLogLoss",
            #    name="mono",
            #    weights=[1,],
            #),
            losses=dict(
                type="DispL1Loss",
                start_disp=0,
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for l1_loss with regard to other loss type
                weight=1,
                # weights for different scale loss
                weights=(0.8 ** 4,),
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="mono_l1_loss"
            ),
        ),
        disp_head=dict(type="PyramidStereoHead2Adabin", # use disp attne conv layer
            in_channels=[192*2], 
            adabins="pixel-wise",
            latent_dims=64,
            ############################################
            max_disp=max_disp,
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
                weights=[  0.8 ** (4 - i - 1) for i in range(1, 5)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        )
)
 
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

work_dir = 'work_dirs/pyramid_stereo_base_weightedloss'

#resume_from = 'work_dirs/pyramid_stereo_base_weightedloss/epoch_2.pth'

find_unused_parameters = True
