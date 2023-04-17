'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-10 22:23:55
LastEditors: Ziming Liu
LastEditTime: 2023-03-11 00:41:55
'''

max_disp = 192
model=dict( 
        type="PSMNet",
        backbone=dict(
            type='context_cluster_medium_feat2', # 64, 128, 320, 512
            style='pytorch',
            num_stages=1,
            out_indices=[0,], # default is [0,2,4,6], there are point reducer layers.
            init_cfg=dict(
                type='Pretrained', 
                checkpoint=\
                    '/home/ziliu/.cache/torch/hub/checkpoints/context_cluster_model_best.pth.tar',
                ),
        ), 
        disp_head=dict(type="PSMNetHead48",
            in_channels=[64*2], 
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
)