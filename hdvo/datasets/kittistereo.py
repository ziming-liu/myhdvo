'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 16:42:40
LastEditors: Ziming Liu
LastEditTime: 2023-07-15 20:09:46
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
class KITTIStereoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, depth_scale_ratio=256, data_prefix=None, test_mode=False, end_id=-1,
                 eval_modality='disp', eval_range=[1,192], filename_tmpl='{:0>10}.png',
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
        print("num samples: ", len(self.video_infos))


    def load_annotations(self):
        print(self.ann_file)
        if isinstance(self.ann_file, (list,tuple)):
            rawdata_list = [mmcv.load(ann_file) for ann_file in self.ann_file]
            data_prefix_list = self.data_prefix
        else:
            rawdata_list = [mmcv.load(self.ann_file)]
            data_prefix_list = [self.data_prefix]
        infos = []
        for data_prefix, rawdata in zip(data_prefix_list, rawdata_list):
            #rawdata = mmcv.load(self.ann_file)
            num_ = len(rawdata)
            #infos = []
            for i in range(num_):
                path = {}
                path['depth_scale_ratio'] = self.depth_scale_ratio
                path["left_frame_paths"] = [os.path.join(data_prefix, rawdata[i]["left_image_path"])]
                path["right_frame_paths"] = [os.path.join(data_prefix,rawdata[i]["right_image_path"])]
                if "left_disp_map_path" in rawdata[i].keys() and  rawdata[i]["left_disp_map_path"] is not None:
                    path["left_disp_paths"] = [os.path.join(data_prefix,rawdata[i]["left_disp_map_path"])]
                    path["left_depth_paths"]= [None]
                elif "left_depth_map_path" in rawdata[i].keys() and rawdata[i]["left_depth_map_path"] is not None: # transfer the depth map to disparity map
                    path["focal"], path["baseline"] = float(rawdata[i]["focal"]), float(rawdata[i]["baseline"])
                    path["left_depth_paths"] = [os.path.join(data_prefix,rawdata[i]["left_depth_map_path"])]
                    path["left_disp_paths"] = [None]
                
                if "right_disp_map_path" in rawdata[i].keys() and rawdata[i]["right_disp_map_path"] is not None:
                    path["right_disp_paths"] = [os.path.join(data_prefix,rawdata[i]["right_disp_map_path"])]
                    path["right_depth_paths"]= [None]
                elif "right_depth_map_path" in rawdata[i].keys() and rawdata[i]["right_depth_map_path"] is not None:
                    path["focal"], path["baseline"] = float(rawdata[i]["focal"]), float(rawdata[i]["baseline"])
                    path["right_depth_paths"] = [os.path.join(data_prefix,rawdata[i]["right_depth_map_path"])]
                    path["right_disp_paths"] = [None]
                infos.append(path)
        return  infos[:self.end_id] if self.end_id > 0 else infos