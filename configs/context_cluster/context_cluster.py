'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-10 23:10:53
LastEditors: Ziming Liu
LastEditTime: 2023-03-11 13:40:34
'''

_base_ = [
    '../_base_/models/context_cluster.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/cascade_stereo_16epoch.py'
]


find_unused_parameters = True

work_dir="work_dirs/context_cluster_base"

