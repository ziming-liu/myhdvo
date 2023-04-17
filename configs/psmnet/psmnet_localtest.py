'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-03-10 16:20:51
'''
_base_ = [
    '../_base_/models/psmnet.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/psmnet_10epoch.py'
]
max_disp = 192
model=dict( 
        type="PSMNet",
        backbone=dict(
            type="PSMNetSingle",
            in_planes=3,  # the in planes of feature extraction backbone
            with_cp=False
        ),
        disp_head=dict(type="PSMNetHead48",
            in_channels=[32*2], 
            local_predictor=True,
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

work_dir = 'work_dirs/psmnet_random256x512'

""" 
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/psmnet/psmnet_localtest.py \
   work_dirs/psmnet_random256x512/epoch_10.pth   \
    --launcher pytorch  --eval  EPE 3PE D1

"""