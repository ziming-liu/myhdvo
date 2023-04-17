'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 18:30:00
LastEditors: Ziming Liu
LastEditTime: 2023-03-12 23:31:56
'''
import os.path as osp

max_disp = 256
model=dict( 
        type="CREStereo",
        max_disp=max_disp,
        mixed_precision=False, 
)

