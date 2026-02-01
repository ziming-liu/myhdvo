import warnings

import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

from hdvo.utils import import_module_error_func
from .registry import BACKBONES, COST_AGGREGATORS, COST_PROCESSORS, DISP_PREDICTORS, HEADS, LOSSES, NECKS, STEREO_PREDICTOR, MASKS, GEOMETRY, MONO_PREDICTOR, VISUAL_ODOMETRY, HYBRID_METHOD


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)

def build_mask(cfg):
    """Build mask."""
    return build(cfg, MASKS)

def build_geometry(cfg):
    """Build geometry modules ."""
    return build(cfg, GEOMETRY)

def build_cost_processor(cfg):
    """Build cost processor modules ."""
    return build(cfg, COST_PROCESSORS)

def build_cost_aggregator(cfg):
    """Build cost aggregator modules ."""
    return build(cfg, COST_AGGREGATORS)

def build_disp_predictor(cfg):
    """Build cost processor modules ."""
    return build(cfg, DISP_PREDICTORS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in STEREO_PREDICTOR:
        return build_stereo_predictor(cfg, train_cfg, test_cfg)
    if obj_type in MONO_PREDICTOR:
        return build_mono_predictor(cfg, train_cfg, test_cfg)
    if obj_type in VISUAL_ODOMETRY:
        return build_visual_odometry(cfg)
    if obj_type in HYBRID_METHOD:
        return build_hybrid_method(cfg)
    raise ValueError(f'{obj_type} is not registered in '
                     'MONO_PREDICTOR, STEREO_PREDICTOR, VISUAL_ODOMETRY, HYBRID_METHOD')


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)

def build_stereo_predictor(cfg,train_cfg=None, test_cfg=None):
    """Build stereo_predictor."""
    return build(cfg, STEREO_PREDICTOR, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_mono_predictor(cfg,train_cfg=None, test_cfg=None):
    """Build monocular_predictor."""
    return build(cfg, MONO_PREDICTOR, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_visual_odometry(cfg):
    """Build visual_odometry."""
    return build(cfg, VISUAL_ODOMETRY)

def build_hybrid_method(cfg):
    """Build hybrid_method."""
    return build(cfg, HYBRID_METHOD)