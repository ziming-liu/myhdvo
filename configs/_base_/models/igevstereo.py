'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 18:30:00
LastEditors: Ziming Liu
LastEditTime: 2023-04-26 18:38:25
'''
import os.path as osp

max_disp = 192
model=dict( 
        type="IGEVStereo",
        backbone=dict(type="IGEVFeatureNet",
        ),
        args=dict(
                restore_ckpt=None,
                mixed_precision=True,
                train_datasets='sceneflow',
                valid_iters=32,
                train_iters=22,
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
        )


)

mixed_precision = True

