'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-10 22:23:55
LastEditors: Ziming Liu
LastEditTime: 2023-03-11 01:04:47
'''
max_disp = 192

model=dict( 
        type="PyramidStereoSceneFlow2_1",
        in_channels=192,
        mask_size=2,
        iters=3,
        backbone=dict(
            type="ConvNeXt",
            depths=[3, 3, 27, 3], 
            dims=[96, 192, 384, 768],
            pretrained="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
            num_stages=2,
            out_indices=[0,1,],
            stem_ratio=2,
        ),
        neck=None,
        mono_head=dict(type="MonoDispHead", 
            max_depth=max_disp//16, 
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
                weights=(1.0, 0.7, 0.5),
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
                weights=(1.0, 1, 1, 1, 1, 1, 1, 1, 1,1.0, 1, 1, 1, 1, 1, 1, 1, 1,1.0, 1, 1, 1, 1, 1, 1, 1, 1,1.0, 1, 1, 1, 1, 1, 1, 1, 1,),
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
                name="stereo_l1_loss"
            ),
        )
)
 