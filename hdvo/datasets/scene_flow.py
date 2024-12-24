'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 16:42:40
LastEditors: Ziming Liu
LastEditTime: 2024-02-07 16:33:00
'''
from ftplib import all_errors
import os.path as osp
from turtle import right
from typing import Sequence
import numpy as np
import random
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
class SceneFlowDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, depth_scale_ratio=None, data_prefix=None, test_mode=False, end_id=-1,
                 eval_modality='disparity', eval_range=[1,100], filename_tmpl='{:0>10}.png',
                   d_filename_tmpl='{:0>10}.png', init_seed=True, **kwargs):
        super().__init__(ann_file=ann_file,
                 pipeline=pipeline,
                 data_prefix=data_prefix,
                 depth_scale_ratio=depth_scale_ratio,
                 test_mode=test_mode,
                 eval_modality=eval_modality,
                 eval_range=eval_range,
                 filename_tmpl=filename_tmpl, 
                 d_filename_tmpl=d_filename_tmpl,
                  init_seed=init_seed, **kwargs)
        self.depth_scale_ratio = depth_scale_ratio
        self.end_id = end_id
        self.pip = pipeline
        self.init_seed = init_seed
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
    

    def evaluate(self,
                 results,
                 gt_labels=None,
                 metrics='EPE',
                 logger=None,
                 **kwargs
                 ):
 
        # Protect ``metric_options`` since it uses mutable value as default 
        if not isinstance(results[0], list):
            raise TypeError(f'results must be a list, but got {type(results[0])}')
        assert len(results[0]) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results[0])} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'EPE', 'D1', '3PE', 'N_DISPS_EPE', 'COUNT_DISP_DIST'
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        predictions = results[0]
        if len(results[1])>0 and gt_labels is None:
            gt_labels = results[1]
        # if there is no gt label in outputs, 
        if gt_labels is None:
            gt_labels = self.get_gt_labels() 
        # if shape is different , can't perform evaluation.
        if gt_labels[0].shape != predictions[0].shape:
            gt_labels = [a[0,...] for a in gt_labels] if len(gt_labels[0].shape)==3 else gt_labels # only keep left view. [batch, Nviews, H, W ]
            predictions = [a[0,...] for a in predictions] if len(predictions[0].shape)==3 else predictions
            assert gt_labels[0].shape == predictions[0].shape, f"the gt and the prediction {gt_labels[0].shape};{predictions[0].shape} should have the smae size."
        # crop to same size ,top left crop
        if gt_labels[0].shape[-2:] != predictions[0].shape[-2:]:
            min_h = min(gt_labels[0].shape[-2], predictions[0].shape[-2])
            min_w = min(gt_labels[0].shape[-1], predictions[0].shape[-1])
            gt_labels = [item.squeeze()[:min_h, :min_w] for item in gt_labels]
            predictions = [item.squeeze()[:min_h, :min_w] for item in predictions]
        out_size = predictions[0].shape[-2:]
        ifstereopad = 0
        for d in self.pip:
            if d['type'] == 'StereoPad':
                ifstereopad = 1
                break
        if ifstereopad:
            print("### using stereo pad, crop the center part of the prediction and gt.")
            predictions = [item.squeeze()[(out_size[0]-540)//2:(out_size[0]-540)//2+540, (out_size[1]-960)//2:(out_size[1]-960)//2+960] for item in predictions]
            gt_labels = [item.squeeze()[(out_size[0]-540)//2:(out_size[0]-540)//2+540, (out_size[1]-960)//2:(out_size[1]-960)//2+960] for item in gt_labels]
        predictions = [item[:, 50:] for item in predictions]
        gt_labels = [item[:, 50:] for item in gt_labels]
        print_copy_log = []
        for metric in metrics:
            #msg = f'Evaluating {metric} ...'
            #if logger is None:
            #    msg = '\n' + msg
            #print_log(msg, logger=logger)

            if metric == 'EPE':
                from ..core.evaluation import epe,epe2
                print("predictions shape ", predictions[0].shape)
                print("gt_labels shape ", gt_labels[0].shape)
                epe_value = epe(predictions, gt_labels, self.eval_range)
                print_copy_log.append(epe_value)
                log_msg = []
                 
                eval_results["EPE"] = epe_value
                log_msg.append(f'EPE: {epe_value:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            
            if metric == 'N_DISPS_EPE':
                from ..core.evaluation import n_disps_epe
                epe_value = n_disps_epe(predictions, gt_labels, self.eval_range)
                epe_value = [float('{:.2f}'.format(i)) for i in epe_value]
                print_copy_log.append(epe_value)
                log_msg = []
                 
                eval_results["N_DISPS_EPE"] = epe_value
                #for i in range(len(epe_value)):
                #    log_msg.append(f'N_DISPS_EPE_{i}: {epe_value[i]:.4f} ')
                log_msg.append(f'from {self.eval_range[0]} to {self.eval_range[1]}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            if metric == 'COUNT_DISP_DIST':
                from ..core.evaluation import count_disp_dist
                epe_value = count_disp_dist(predictions, gt_labels, self.eval_range)
                epe_value = [float('{:.2f}'.format(i)) for i in epe_value]
                print_copy_log.append(epe_value)
                log_msg = []
                 
                eval_results["COUNT_DISP_DIST"] = epe_value
                #for i in range(len(epe_value)):
                #    log_msg.append(f'N_DISPS_EPE_{i}: {epe_value[i]:.4f} ')
                log_msg.append(f'from {self.eval_range[0]} to {self.eval_range[1]}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == '3PE':
                from ..core.evaluation import three_pe
                three_pe_value = three_pe(predictions, gt_labels, self.eval_range)
                print_copy_log.append(three_pe_value)
                log_msg = []
                 
                eval_results["3PE"] = three_pe_value
                log_msg.append(f'3PE: {three_pe_value:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            
            if metric == 'D1':
                from ..core.evaluation import d1
                d1_value = d1(predictions, gt_labels, self.eval_range)
                print_copy_log.append(d1_value)
                log_msg = []
                 
                eval_results["D1"] = d1_value
                log_msg.append(f'D1: {d1_value:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
        for i in range(len(print_copy_log)):
            print(print_copy_log[i])
        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # prepare tensor in getitem
        if self.depth_scale_ratio is not None:
            results["depth_scale_ratio"] = self.depth_scale_ratio
        results['filename_tmpl'] = self.filename_tmpl
        results['d_filename_tmpl'] = self.d_filename_tmpl
        results['eval_modality'] = self.eval_modality
        #results['pose'] = self._process_gt_pose(results['pose'])

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # prepare tensor in getitem
        if self.depth_scale_ratio is not None:
            results["depth_scale_ratio"] = self.depth_scale_ratio
        results['filename_tmpl'] = self.filename_tmpl
        results['d_filename_tmpl'] = self.d_filename_tmpl
        results['eval_modality'] = self.eval_modality
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            #print(worker_info)
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                #print(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
