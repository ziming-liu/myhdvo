'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 16:42:40
LastEditors: Ziming Liu
LastEditTime: 2023-07-20 01:02:01
'''
from ftplib import all_errors
import os.path as osp
from turtle import right
from typing import Sequence
import numpy as np
import cv2
import json
import argparse
from tqdm import tqdm
from .evaluation_utils import *
import os
import glob
from sys import prefix
import torch
import pandas as pd
import  time
import numpy as np
from tqdm import tqdm
import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from hdvo.utils import register_module_hooks, get_root_logger
import mmcv 

from ..core.geometry.camera_modules import Intrinsics

from .base import BaseDataset
from .registry import DATASETS
 


@DATASETS.register_module()
class MultiStereoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, depth_scale_ratio,
                 data_prefix=None, test_mode=False, end_id=-1,
                 eval_modality='disparity', eval_range=[1,192], filename_tmpl='{:0>10}.png',
                   d_filename_tmpl='{:0>10}.png', **kwargs):
        super().__init__(ann_file=ann_file,
                 pipeline=pipeline,
                 data_prefix=data_prefix,
                 depth_scale_ratio=depth_scale_ratio,
                 test_mode=test_mode,
                 eval_modality=eval_modality,
                 eval_range=eval_range,
                 filename_tmpl=filename_tmpl, 
                 d_filename_tmpl=d_filename_tmpl, )
        self.data_prefix = data_prefix
        self.depth_scale_ratio = depth_scale_ratio
        self.end_id = end_id
        self.video_infos = self.load_annotations()

    def load_annotations(self):
        if isinstance(self.ann_file, (list,tuple)):
            all_depth_scale_ratio = []
            all_data_prefix = []
            rawdata = []
            for i, ann_file in enumerate(self.ann_file):
                data = mmcv.load(ann_file)
                all_depth_scale_ratio.extend([ self.depth_scale_ratio[i] for _ in range(len(data))])
                all_data_prefix.extend([ self.data_prefix[i] for _ in range(len(data))])
                rawdata.extend(data)

        else:
            rawdata = mmcv.load(self.ann_file)
            all_depth_scale_ratio = [self.depth_scale_ratio for _ in range(len(rawdata))]
            all_data_prefix = [self.data_prefix for _ in range(len(rawdata))]
        num_ = len(rawdata)
        infos = []
        for i in range(num_):
            path = {}
            path['depth_scale_ratio'] = all_depth_scale_ratio[i]
            path["left_frame_paths"] = [os.path.join(all_data_prefix[i], rawdata[i]["left_image_path"])]
            path["right_frame_paths"] = [os.path.join(all_data_prefix[i],rawdata[i]["right_image_path"])]
            if "left_disp_map_path" in rawdata[i] and rawdata[i]["left_disp_map_path"] is not None:
                path["left_disp_paths"] = [os.path.join(all_data_prefix[i],rawdata[i]["left_disp_map_path"])]
            if "right_disp_map_path" in rawdata[i] and rawdata[i]["right_disp_map_path"] is not None:
                path["right_disp_paths"] = [os.path.join(all_data_prefix[i],rawdata[i]["right_disp_map_path"])]
            if "left_depth_map_path" in rawdata[i] and rawdata[i]["left_depth_map_path"] is not None:
                path["left_depth_paths"] = [os.path.join(all_data_prefix[i],rawdata[i]["left_depth_map_path"])]
            if "right_depth_map_path" in rawdata[i] and rawdata[i]["right_depth_map_path"] is not None:
                path["right_depth_paths"] = [os.path.join(all_data_prefix[i],rawdata[i]["right_depth_map_path"])]
            if "focal" in rawdata[i]:
                path['focal'] = float(rawdata[i]["focal"])
            if "baseline" in rawdata[i]:
                path["baseline"] = float(rawdata[i]["baseline"])
            infos.append(path)
        return  infos[:self.end_id] if self.end_id > 0 else infos