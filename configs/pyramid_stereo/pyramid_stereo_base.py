'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-11 00:44:06
LastEditors: Ziming Liu
LastEditTime: 2023-03-11 13:36:39
'''
_base_ = [
    '../_base_/models/pyramid_stereo.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/cascade_stereo_16epoch.py'
]


work_dir = 'work_dirs/pyramid_stereo_base'

