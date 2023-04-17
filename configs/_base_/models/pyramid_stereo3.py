'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-22 13:54:22
LastEditors: Ziming Liu
LastEditTime: 2023-03-22 19:36:54
'''
 
max_disp = 192
iters = 4
model=dict( 
        type="PyramidStereoSceneFlow3",
        in_channels=192,
        mask_size=4,
        iters=iters,
        search_ranges=[12,12,12],
        max_disp=max_disp,
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
        mono_head=None,
        disp_head=[dict(type="PSMNetHead48Adabin", # use disp attne conv layer
            in_channels=[192*2], 
            ############################################
            disp_range=[0,max_disp//16,1], # [min, max, step]
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
                weights=[  0.8 ** (2*iters - i - 1) for i in range(0, 2*iters)],
                sparse=False,
                set_range=True,
                random_mask = False, 
                random_mask_ratio = 0.5,
            ),
        ),
        dict(type="PSMNetHead48Adabin", # use disp attne conv layer
            in_channels=[192*2], 
            ############################################
            disp_range=[0,max_disp//8,1], # [min, max, step]
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
        dict(type="PSMNetHead48Adabin", # use disp attne conv layer
            in_channels=[192*2], 
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
        )
        ]

)
 