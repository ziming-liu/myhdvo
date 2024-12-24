'''
Author: Ziming Liu
Date: 2021-04-25 13:09:26
LastEditors: Ziming Liu
LastEditTime: 2023-06-24 00:12:18
Description: ...
Dependent packages: don't need any extral dependency
'''
from mmcv.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
MASKS =  Registry('mask')
GEOMETRY =  Registry('geometry')
COST_PROCESSORS= Registry('cost_processor')
DISP_PREDICTORS = Registry('disp_predictor')
COST_AGGREGATORS = Registry('cost_aggregator')
MONO_PREDICTOR = Registry('mono_predictor')
STEREO_PREDICTOR = Registry('stereo_predictor')
VISUAL_ODOMETRY = Registry('visual_odometry')
HYBRID_METHOD = Registry('hybrid_method')
