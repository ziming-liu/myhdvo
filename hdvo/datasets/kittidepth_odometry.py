"""
KITTI Odometry Dataset for depth and pose estimation.

This module provides a dataset loader for the KITTI Odometry benchmark,
supporting stereo depth estimation and visual odometry evaluation.
"""

import os
import random
import time
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm
from mmcv.utils import print_log
import mmcv

from .base import BaseDataset
from .registry import DATASETS

@DATASETS.register_module()
class KITTIOdometryDataset(BaseDataset):
    """KITTI Odometry Dataset for stereo depth and pose estimation.
    
    Args:
        ann_file (str): Path to annotation file.
        pipeline (list): Data processing pipeline.
        depth_scale_ratio (int): Scale ratio for depth values. Default: 256.
        data_prefix (str): Prefix of data path. Default: None.
        test_mode (bool): Whether in test mode. Default: False.
        end_id (int): End index for loading samples (-1 for all). Default: -1.
        eval_modality (str): Evaluation modality. Default: 'disparity'.
        eval_range (list): Evaluation range. Default: [1, 192].
        filename_tmpl (str): Template for image filenames. Default: '{:0>10}.png'.
        d_filename_tmpl (str): Template for depth filenames. Default: '{:0>10}.png'.
        crop_test_image (str): Crop strategy for test images. Default: 'garg'.
        camera (str): Camera pair to use ('01' or '23'). Default: '23'.
        test_seq_id (int): Sequence ID for testing. Default: 99.
        load_gtdepth (bool): Whether to load ground truth depth. Default: False.
        kitti_rawdata_path (str): Path to KITTI raw data. Default: None.
    """
    
    def __init__(self, ann_file, pipeline, depth_scale_ratio=256, data_prefix=None, 
                 test_mode=False, end_id=-1, eval_modality='disparity', eval_range=[1, 192], 
                 filename_tmpl='{:0>10}.png', d_filename_tmpl='{:0>10}.png', 
                 crop_test_image='garg', camera="23", test_seq_id=99, load_gtdepth=False, 
                 kitti_rawdata_path=None, **kwargs):
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            depth_scale_ratio=depth_scale_ratio,
            test_mode=test_mode,
            eval_modality=eval_modality,
            eval_range=eval_range,
            filename_tmpl=filename_tmpl, 
            d_filename_tmpl=d_filename_tmpl
        )
        self.kitti_rawdata_path = kitti_rawdata_path
        self.test_seq_id = test_seq_id
        self.camera = camera
        self.end_id = end_id
        self.test_mode = test_mode
        self.crop_test_image = crop_test_image
        self.load_gtdepth = load_gtdepth
        self.video_infos = self.load_annotations()
        print(f"Loaded {len(self.video_infos)} samples")


    def load_annotations(self):
        """Load annotations from annotation file.
        
        Returns:
            list: List of video information dictionaries.
        """
        rawdata = mmcv.load(self.ann_file)
        num_ = len(rawdata)
        infos = []
        
        for i in range(num_):
            path = {}
            
            # Skip sequence 03 if loading ground truth depth (sequence 03 has missing GT depth)
            if self.load_gtdepth and "/03/" in rawdata[i]["image_2_paths"][0]:
                continue
            
            # Load paths and intrinsics based on camera pair
            if self.camera == "01":
                path["left_frame_paths"] = [
                    os.path.join(self.data_prefix, *a.split('/')[7:]) 
                    for a in rawdata[i]["image_0_paths"]
                ]
                path["right_frame_paths"] = [
                    os.path.join(self.data_prefix, *a.split('/')[7:]) 
                    for a in rawdata[i]["image_1_paths"]
                ]
                path["k_left"] = np.array([float(x) for x in rawdata[i]["K_0"].strip().split(' ')]).reshape(3, 4)[:3, :3].astype(np.float32)
                path["K_right"] = np.array([float(x) for x in rawdata[i]["K_1"].strip().split(' ')]).reshape(3, 4)[:3, :3].astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_0"])
                path["focal_right"] = float(rawdata[i]["focal_1"])
                path["baseline"] = float(rawdata[i]["baseline_01"])
                
            elif self.camera == "23":
                path["left_frame_paths"] = [
                    os.path.join(self.data_prefix, *a.split('/')[7:]) 
                    for a in rawdata[i]["image_2_paths"]
                ]
                path["right_frame_paths"] = [
                    os.path.join(self.data_prefix, *a.split('/')[7:]) 
                    for a in rawdata[i]["image_3_paths"]
                ]
                path["k_left"] = np.array([float(x) for x in rawdata[i]["K_2"].strip().split(' ')]).reshape(3, 4)[:3, :3].astype(np.float32)
                path["K_right"] = np.array([float(x) for x in rawdata[i]["K_3"].strip().split(' ')]).reshape(3, 4)[:3, :3].astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_2"])
                path["focal_right"] = float(rawdata[i]["focal_3"])
                path["baseline"] = float(rawdata[i]["baseline_23"])
                
                # Load ground truth depth paths if needed
                if self.load_gtdepth:
                    path["left_depth_paths"] = [
                        p.replace("image_2", "depth_2") 
                        for p in path["left_frame_paths"]
                    ]
                    path["right_depth_paths"] = [
                        p.replace("image_3", "depth_3") 
                        for p in path["right_frame_paths"]
                    ]
                    path["depth_scale_ratio"] = self.depth_scale_ratio
            
            # Verify intrinsics consistency
            assert (path["k_left"] == path["K_right"]).all()
            assert path['focal'] == path["focal_right"]
            
            path['intrinsics'] = np.stack([path["k_left"], path["K_right"]], 0)
            
            # Load ground truth poses if available
            if "gt_poses" in rawdata[0] and rawdata[0]["gt_poses"] is not None:
                pose = [
                    np.array([float(x) for x in a.strip().split(' ')]).reshape(3, 4) 
                    for a in rawdata[i]["gt_poses"]
                ]
                pose = [np.concatenate([a, np.array([[0, 0, 0, 1]])], 0) for a in pose]
                path["pose"] = np.stack(pose).astype(np.float32)
            
            self.seq_dir = '/'.join(path["left_frame_paths"][0].split("/")[:-2])
            infos.append(path)
        
        if self.end_id != -1:
            return infos[:self.end_id]
        else:
            return infos


