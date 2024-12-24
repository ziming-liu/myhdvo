'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 16:42:40
LastEditors: Ziming Liu
LastEditTime: 2023-07-15 20:12:13
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
class CREStereoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False, end_id=-1,depth_scale_ratio=32,
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
        self.depth_scale_ratio = depth_scale_ratio
        self.end_id = end_id
        self.video_infos = self.load_annotations()

    def load_annotations(self):

        rawdata = mmcv.load(self.ann_file)
        num_ = len(rawdata)
        infos = []
        for i in range(num_):
            path = {}
            path['depth_scale_ratio'] = self.depth_scale_ratio
            path["left_frame_paths"] = [os.path.join(self.data_prefix, rawdata[i]["left_image_path"])]
            path["right_frame_paths"] = [os.path.join(self.data_prefix,rawdata[i]["right_image_path"])]
            path["left_disp_paths"] = [os.path.join(self.data_prefix,rawdata[i]["left_disp_map_path"])]
            path["right_disp_paths"] = [os.path.join(self.data_prefix,rawdata[i]["right_disp_map_path"])]
            infos.append(path)
        return  infos[:self.end_id]