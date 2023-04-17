'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:57:53
LastEditors: Ziming Liu
LastEditTime: 2023-03-10 02:05:08
'''
_base_ = [
    '../_base_/models/psmnet.py',
    '../_base_/datasets/scene_flow_random256x320.py', '../_base_/default_runtime.py',
    '../_base_/schedules/psmnet_10epoch.py'
]


work_dir = 'work_dirs/psmnet_random256x320'