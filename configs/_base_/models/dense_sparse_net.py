'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:46:30
LastEditors: Ziming Liu
LastEditTime: 2023-04-01 00:55:30
'''

import os.path as osp

max_disp = 192
model=dict( 
        type="PSMNet",
        backbone=dict(
            type="PSMNetSingle",
            in_planes=3,  # the in planes of feature extraction backbone
            with_cp=False
        ),
        disp_head=dict(type="DenseSparseHead",
            in_channels=[32*2], 
            ############################################
            dense_disp_range=[0,48,1], # [min, max, step] # same as the W/4
            sparse_disp_range=[0,48,6], 
            pred_disp_range=[0,192,1], # [min, max, step]
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