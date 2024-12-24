'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 16:42:40
LastEditors: Ziming Liu
LastEditTime: 2023-08-14 16:51:18
'''
from ftplib import all_errors
import os.path as osp
from turtle import right
from typing import Sequence
import numpy as np
from mmcv.utils import print_log

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
class KITTIDepthEigenDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, depth_scale_ratio=256, data_prefix=None, test_mode=False, end_id=-1,
                 eval_modality='disparity', eval_range=[1,192], filename_tmpl='{:0>10}.png',
                   d_filename_tmpl='{:0>10}.png', crop_test_image='garg', camera="23", **kwargs):
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
        self.camera = camera
        self.pip = pipeline
        self.end_id = end_id
        self.test_mode = test_mode
        self.crop_test_image = crop_test_image
        if not test_mode:
            self.video_infos = self.load_annotations()
        else:
            self.video_infos = self.load_annotations_test()
        print("num samples: ", len(self.video_infos))

    def load_annotations(self):
        rawdata = mmcv.load(self.ann_file)
        num_ = len(rawdata)
        infos = []
        for i in range(num_):
            path = {}
            path['depth_scale_ratio'] = self.depth_scale_ratio
            if self.camera=="01":
                path["left_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_0_paths"]]
                path["right_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_1_paths"]]
                path["k_left"] = np.array(rawdata[i]["K_0"]).reshape(3,3).astype(np.float32)
                path["K_right"] = np.array(rawdata[i]["K_1"]).reshape(3,3).astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_0"])
                path["focal_right"] = float(rawdata[i]["focal_1"])
                path["baseline"] = float(rawdata[i]["baseline_01"])
            elif self.camera=="23":
                
                path["left_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_2_paths"]] 
                path["right_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_3_paths"]]
                path["k_left"] = np.array(rawdata[i]["K_2"]).reshape(3,3).astype(np.float32)
                path["K_right"] = np.array(rawdata[i]["K_3"]).reshape(3,3).astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_2"])
                path["focal_right"] = float(rawdata[i]["focal_3"])
                path["baseline"] = float(rawdata[i]["baseline_23"])
            assert (path["k_left"] == path["K_right"]).all()
            assert (path['focal'] == path["focal_right"] )
            path['intrinsics'] = np.stack([path["k_left"], path["K_right"]],0)
            path["pose"] = np.stack([np.array(a).reshape(4,4) for a in rawdata[i]["gt_poses"]]).astype(np.float32)
            
            infos.append(path)
        return  infos[:self.end_id] if self.end_id!=-1 else infos
    
    def load_annotations_test(self):
        rawdata = mmcv.load(self.ann_file)
        num_ = len(rawdata)
        infos = []
        for i in range(num_):
            path = {}
            if "gt_poses" in rawdata[i] and len(rawdata[i]["gt_poses"])>0:
                path["pose"] = np.stack([np.array(a).reshape(4,4) for a in rawdata[i]["gt_poses"]]).astype(np.float32)
            if self.camera=="01":
                path["left_frame_paths"] = [a for a in rawdata[i]["image_0_paths"]]
                path["right_frame_paths"] = [a for a in rawdata[i]["image_1_paths"]]
                path["k_left"] = np.array(rawdata[i]["K_0"]).reshape(3,3).astype(np.float32)
                path["K_right"] = np.array(rawdata[i]["K_1"]).reshape(3,3).astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_0"])
                path["focal_right"] = float(rawdata[i]["focal_1"])
                path["baseline"] = float(rawdata[i]["baseline_01"])
            elif self.camera=="23":
                path["left_frame_paths"] = [a for a in rawdata[i]["image_2_paths"]] 
                path["right_frame_paths"] = [a for a in rawdata[i]["image_3_paths"]]
                path["k_left"] = np.array(rawdata[i]["K_2"]).reshape(3,3).astype(np.float32)
                path["K_right"] = np.array(rawdata[i]["K_3"]).reshape(3,3).astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_2"])
                path["focal_right"] = float(rawdata[i]["focal_3"])
                path["baseline"] = float(rawdata[i]["baseline_23"])
            assert (path["k_left"] == path["K_right"]).all()
            assert (path['focal'] == path["focal_right"] )
            path['intrinsics'] = np.stack([path["k_left"], path["K_right"]],0)
            infos.append(path)
        return  infos[:self.end_id]
    
    def evaluate(self, results, gt_labels=None, metrics='EPE', logger=None, **kwargs):
    
        split =  'eigen'
        if split=='kitti':
            gt_path = '/data/acentauri/user/ziliu/data/stereo_matching_data/StereoMatching/KITTI-2015/'
        elif split =='eigen':
            gt_path = '/data/acentauri/user/ziliu/data/kitti_raw_data/'
        else:
            raise ValueError
        print("### start to evaluate KITTI DEPTH, {} split ###".format(split))
        min_depth = 1 #2.018
        max_depth = 80
        if self.crop_test_image is None:
            garg_crop, eigen_crop = False, False
        elif  self.crop_test_image == "eigen":
            garg_crop, eigen_crop = False, True
        elif self.crop_test_image == "garg":
            garg_crop, eigen_crop = True, False
        print(f">> test image with {self.crop_test_image} crop")
        V = results[0][0].shape[0]
        for view_idx in range(V):
            print("### FOR the view ", view_idx)
            pred_disparities = [results[0][bi][view_idx,:,:] for bi in range(len(results[0]))] # shape: batchsize, h, w # keep view 0 dimension 
            frame_dir = results[-1]
        
            if split == 'kitti':
                num_samples = 200
                
                gt_disparities = load_gt_disp_kitti(gt_path)
                gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

            elif split == 'eigen':
                num_samples = 697
                #test_files = read_text_lines('/data/acentauri/user/ziliu/data/depth_splits/eigen_test_files.txt')
                #test_files = read_text_lines(self.ann_file)
                #test_files = read_text_lines('/data/acentauri/user/ziliu/data/depth_splits/eigen_monodepth/eigen_test_files.txt')
                rawdata = mmcv.load(self.ann_file)
                if self.camera == "01":
                    test_files = ['/'.join(a["image_0_paths"][0].split('/')[-5:]) for a in rawdata]
                elif self.camera == "23":
                    test_files = ['/'.join(a["image_2_paths"][0].split('/')[-5:]) for a in rawdata]
                test_files = test_files[:len(pred_disparities)] 
                gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path)

                num_test = len(im_files)
                gt_depths = []
                pred_depths = []
                print("# load gt depth")
                if len(pred_disparities) < num_samples:
                    num_samples = len(pred_disparities)
                for t_id in tqdm(range(num_samples)):
                    camera_id = cams[t_id]  # 2 is left, 3 is right
                    if self.camera == "01": camera_id = 0 # TODO: support 01 camera evaluation
                    #print("gt calib {}".format(gt_calib[t_id]))
                    depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
                    kbcrop = False
                    for d in self.pip:
                        if d['type'] == 'KITTIKBCrop':
                            crop_size = d['crop_size']
                            kbcrop = True
                    if kbcrop:
                        gth, gtw = im_sizes[t_id]
                        assert len(depth.shape)==2
                        depth = depth[gth-crop_size[0]:gth,(gtw-crop_size[1])//2:(gtw-crop_size[1])//2+crop_size[1]]
                    
                    gt_depths.append(depth.astype(np.float32))
                    # im_sizes: (h, w)
                    
                    if kbcrop: # if use kbcrop test, we will not use resize test 
                        disp_pred = pred_disparities[t_id].squeeze()
                    else:
                        disp_pred = cv2.resize(pred_disparities[t_id].squeeze(), (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_NEAREST)
                    assert disp_pred.shape == depth.shape, "depth shape {}, disp shape {}".format(depth.shape, disp_pred.shape)
                    #print("origin disp \n {}".format(disp_pred))
                    #TODO: why x width, scale disparity for monodepth?? acfnet don;t need this. 
                    #disp_pred = disp_pred * disp_pred.shape[1]
                    #print("x width disp \n {}".format(disp_pred))
                    # need to convert from disparity to depth
                    #focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
                    #depth_pred = (baseline * focal_length) / disp_pred
                    # cm to m 
                    depth_pred = disp_pred # for hybrid vo return, the return disp_pred is already depth.
                    depth_pred[np.isinf(depth_pred)] = 0
                    #print("depth pred  \n {}".format(depth_pred))
                    pred_depths.append(depth_pred)
                    
                    #assert frame_dir[t_id] == '/'.join(im_files[t_id].split('/')[:-3]), "{}".format('/'.join(im_files[t_id].split('/')[:-3]))
            pred_depths = pred_depths[:num_samples]
            assert len(pred_depths) == len(gt_depths)
            rms     = np.zeros(num_samples, np.float32)
            log_rms = np.zeros(num_samples, np.float32)
            abs_rel = np.zeros(num_samples, np.float32)
            sq_rel  = np.zeros(num_samples, np.float32)
            d1_all  = np.zeros(num_samples, np.float32)
            a1      = np.zeros(num_samples, np.float32)
            a2      = np.zeros(num_samples, np.float32)
            a3      = np.zeros(num_samples, np.float32)
            print("# compute metrics...")
            for i in tqdm(range(num_samples)):
                
                gt_depth = gt_depths[i]
                pred_depth = pred_depths[i]
                pred_depth[pred_depth < min_depth] = min_depth
                pred_depth[pred_depth > max_depth] = max_depth

                if split == 'eigen':
                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

                    if garg_crop or eigen_crop:
                        gt_height, gt_width = gt_depth.shape

                        # crop used by Garg ECCV16
                        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                        if garg_crop:
                            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                            0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                        # crop we found by trial and error to reproduce Eigen NIPS14 results
                        elif eigen_crop:
                            crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                            0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                        crop_mask = np.zeros(mask.shape)
                        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                        mask = np.logical_and(mask, crop_mask)

                if split == 'kitti':
                    gt_disp = gt_disparities[i]
                    mask = gt_disp > 0
                    pred_disp = pred_disparities_resized[i]

                    disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
                    bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
                    d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

                abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
            
            msg = f'Evaluating  view {view_idx}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
            print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean()*100, a2.mean()*100, a3.mean()*100))
            eval_res = {'abs_rel':abs_rel.mean(), 'sq_rel': sq_rel.mean(), 'rms':rms.mean(), 'log_rms': log_rms.mean(), 'd1_all':d1_all.mean(), 'a1':a1.mean()*100, 'a2':a2.mean()*100, 'a3': a3.mean()*100}
            log_msg = []
            log_msg.append(f'\n evaluation results')
            for k, acc in eval_res.items():
                #depth_eval_results[f'_{k}'] = acc
                log_msg.append(f'  {k}: {acc:.4f}  ')
                #self.logger.info(f'  {k}: {acc:.4f}  ')
            log_msg = ' '.join(log_msg)
            print_log(log_msg, logger=logger)