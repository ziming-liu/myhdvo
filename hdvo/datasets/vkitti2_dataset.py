'''
VKitti2 Dataset Loader for Visual Odometry

This module provides a dataset loader for the Virtual KITTI 2 (VKitti2) dataset,
designed for stereo visual odometry tasks.

Authors: ACENTAURI team, INRIA
License: See LICENSE file in the root directory
'''
import os.path as osp
import os
import numpy as np
from mmcv.utils import print_log
import random
import cv2
import glob
import torch
import pandas as pd
import time
import copy
import warnings
from tqdm import tqdm

import mmcv 

from .base import BaseDataset
from .registry import DATASETS
from hdvo.models.utils.pose_utils import pose_relative2absolute


def load_vkitti2_odom_intrinsics(camera_intrinsic_path, new_h, new_w):
    """Load virtual kitti2 odometry data intrinsics
    frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
    Args:
        camera_intrinsic_path (str): txt file path
        new_h (int): target image height
        new_w (int): target image width
    
    Returns:
        intrinsics (dict): each element contains [cx, cy, fx, fy]
        K_left (np.array): 3x3 intrinsic matrix for left camera
        K_right (np.array): 3x3 intrinsic matrix for right camera
    """
    assert new_h < new_w
    raw_img_h = 375.0  # VKitti2: 1242 x 375
    raw_img_w = 1242.0
    intrinsics = {}  # only save intri from left camera, because they are the same.
    
    with open(camera_intrinsic_path, 'r') as cf:
        raw_intrisic = cf.readlines()
        raw_intrisic = raw_intrisic[1:]  # remove heading in txt
        left_cam_intri = raw_intrisic[::2]
        right_cam_intri = raw_intrisic[1::2]
        assert len(left_cam_intri) == len(right_cam_intri)
        
        for i, item in enumerate(left_cam_intri):
            line_split = [float(value) for value in item.strip().split(' ')[2:]]
            assert len(line_split) == 4
            intrinsics[i] = [
                line_split[2] / raw_img_w * new_w,  # cx 
                line_split[3] / raw_img_h * new_h,  # cy
                line_split[0] / raw_img_w * new_w,  # fx
                line_split[1] / raw_img_h * new_h,  # fy
            ]
    
    # Get first frame intrinsics (they're constant in VKitti2)
    first_intri = intrinsics[0]
    cx, cy, fx, fy = first_intri
    
    # Build 3x3 intrinsic matrices
    K_left = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    K_right = K_left.copy()  # Same for both cameras in VKitti2
    
    return intrinsics, K_left, K_right


@DATASETS.register_module()
class VKitti2Dataset(BaseDataset):
    """
    VKitti2 Dataset for stereo visual odometry.
    
    VKitti2 structure:
    - Scene01/clone/frames/
        - rgb/Camera_0/*.jpg (left camera)
        - rgb/Camera_1/*.jpg (right camera)
        - depth/Camera_0/*.png
        - depth/Camera_1/*.png
    - Scene01/clone/intrinsic.txt
    - Scene01/clone/extrinsic.txt
    - Scene01/clone/pose.txt
    
    Camera intrinsics for VKitti2 (default at 1242x375):
    - focal: 725.0087
    - cx: 620.5
    - cy: 187.0
    - baseline: 0.532725 (meters)
    """
    
    # VKitti2 default camera parameters (at original resolution 1242x375)
    RAW_IMG_WIDTH = 1242.0
    RAW_IMG_HEIGHT = 375.0
    BASELINE = 0.532725  # meters
    
    def __init__(self, ann_file=None, ann_files=None, pipeline=None, depth_scale_ratio=100, data_prefix=None, 
                 test_mode=False, end_id=-1, eval_modality='depth', eval_range=[1, 80],
                 filename_tmpl='rgb_{:0>5}.jpg', d_filename_tmpl='depth_{:0>5}.png',
                 crop_test_image=None, camera="01", test_seq_id=None, 
                 load_gtdepth=False, scene='Scene01', variation='clone', 
                 target_size=(1024, 320), **kwargs):
        """
        Args:
            ann_file: Path to single annotation file (json format) - for backward compatibility
            ann_files: List of paths to annotation files (json format) - for multi-scene training
            pipeline: Data processing pipeline
            depth_scale_ratio: Scale factor for depth (VKitti2 uses 100)
            data_prefix: Root directory of VKitti2 dataset
            test_mode: Whether in test mode
            end_id: End index for limiting dataset size (-1 for all)
            eval_modality: 'depth' or 'disparity'
            eval_range: [min_depth, max_depth] for evaluation
            filename_tmpl: Template for RGB image filenames
            d_filename_tmpl: Template for depth image filenames
            crop_test_image: Crop strategy for test images
            camera: "01" for Camera_0/Camera_1
            test_seq_id: Sequence ID for testing
            load_gtdepth: Whether to load ground truth depth
            scene: Scene name (e.g., 'Scene01', 'Scene02')
            variation: Scene variation (e.g., 'clone', '15-deg-left', 'fog')
            target_size: (width, height) for resizing images
        """
        # Support both single ann_file and multiple ann_files
        if ann_files is not None:
            self.ann_files = ann_files if isinstance(ann_files, list) else [ann_files]
            ann_file = self.ann_files[0]  # Use first file for parent class
        elif ann_file is not None:
            self.ann_files = [ann_file]
        else:
            raise ValueError("Either ann_file or ann_files must be provided")
        
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            depth_scale_ratio=depth_scale_ratio,
            test_mode=test_mode,
            eval_modality=eval_modality,
            eval_range=eval_range,
            filename_tmpl=filename_tmpl, 
            d_filename_tmpl=d_filename_tmpl,
        )
        
        self.scene = scene
        self.variation = variation
        self.test_seq_id = test_seq_id if test_seq_id is not None else f"{scene}_{variation}"
        self.camera = camera
        self.end_id = end_id
        self.test_mode = test_mode
        self.crop_test_image = crop_test_image
        self.load_gtdepth = load_gtdepth
        self.target_size = target_size  # (width, height)
        
        # Load annotations
        self.video_infos = self.load_annotations()
        print_log(f"Loaded {len(self.video_infos)} sequences from VKitti2 "
                 f"({len(self.ann_files)} annotation file(s))", logger='current')

    def load_annotations(self):
        """Load annotations from JSON file(s) or scan directory."""
        infos = []
        
        # Load from multiple annotation files
        for ann_file in self.ann_files:
            if osp.exists(ann_file):
                print_log(f"Loading annotations from {ann_file}", logger='current')
                rawdata = mmcv.load(ann_file)
                num_ = len(rawdata)
                
                # Try to extract scene/variation from annotation file path or use defaults
                # Example path: "annotations/vkitti2/weather/clone/scene01.json"
                current_scene = self.scene
                current_variation = self.variation
                
                # Try to infer scene and variation from file path
                path_parts = ann_file.split('/')
                if 'vkitti2' in path_parts:
                    vkitti2_idx = path_parts.index('vkitti2')
                    # Check if there's enough path structure after vkitti2
                    if len(path_parts) > vkitti2_idx + 2:
                        # Possible structure: .../vkitti2/weather/clone/scene01.json
                        potential_variation = path_parts[vkitti2_idx + 2] if path_parts[vkitti2_idx + 1] in ['weather', 'Weather'] else path_parts[vkitti2_idx + 1]
                        potential_scene = path_parts[-1].replace('.json', '')
                        
                        # Validate and use if reasonable
                        if potential_scene.lower().startswith('scene'):
                            # Capitalize to match directory structure: Scene01, Scene02, etc.
                            current_scene = potential_scene.capitalize() if not potential_scene[0].isupper() else potential_scene
                            if 'Scene' not in current_scene and 'scene' in current_scene.lower():
                                current_scene = 'Scene' + current_scene.lower().replace('scene', '')
                        
                        if potential_variation in ['clone', '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 
                                                   'fog', 'morning', 'overcast', 'rain', 'sunset']:
                            current_variation = potential_variation
                
                print_log(f"Processing scene={current_scene}, variation={current_variation}", logger='current')
                
                # Load intrinsics from VKitti2 intrinsic.txt file
                intrinsic_path = osp.join(self.data_prefix, current_scene, current_variation, 'intrinsic.txt')
                if osp.exists(intrinsic_path):
                    _, K_left, K_right = load_vkitti2_odom_intrinsics(
                        intrinsic_path, 
                        self.target_size[1],  # height
                        self.target_size[0]   # width
                    )
                    focal_length = K_left[0, 0]
                else:
                    raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}. "
                                          f"Please ensure VKitti2 dataset is properly structured.")
                
                for i in range(num_):
                    path = {}
                    
                    # Left and right image paths
                    if self.camera == "01":
                        path["left_frame_paths"] = rawdata[i].get("image_0_paths", rawdata[i].get("left_paths", []))
                        path["right_frame_paths"] = rawdata[i].get("image_1_paths", rawdata[i].get("right_paths", []))
                    else:
                        raise ValueError(f"Unsupported camera configuration: {self.camera}")
                    
                    # Use loaded camera intrinsics
                    path["k_left"] = K_left
                    path["K_right"] = K_right
                    path['focal'] = focal_length
                    path["focal_right"] = focal_length
                    path["baseline"] = self.BASELINE
                    path['intrinsics'] = np.stack([K_left, K_right], 0)
                    
                    # Load depth paths if needed
                    if self.load_gtdepth and "depth_0_paths" in rawdata[i]:
                        path["left_depth_paths"] = rawdata[i]["depth_0_paths"]
                        path["right_depth_paths"] = rawdata[i].get("depth_1_paths", [])
                        path["depth_scale_ratio"] = self.depth_scale_ratio
                    
                    # Load ground truth poses if available
                    if "gt_poses" in rawdata[i] and rawdata[i]["gt_poses"] is not None:
                        poses = []
                        for pose_str in rawdata[i]["gt_poses"]:
                            if isinstance(pose_str, str):
                                pose_vals = np.array([float(x) for x in pose_str.strip().split()])
                                if len(pose_vals) == 12:
                                    # Reshape 3x4 to 4x4
                                    pose = pose_vals.reshape(3, 4)
                                    pose = np.vstack([pose, [0, 0, 0, 1]])
                                else:
                                    pose = pose_vals.reshape(4, 4)
                            else:
                                pose = np.array(pose_str)
                            poses.append(pose.astype(np.float32))
                        path["pose"] = np.stack(poses)
                    
                    infos.append(path)
            else:
                raise FileNotFoundError(f"Annotation file not found: {ann_file}. "
                                      f"Please check the path and ensure the file exists.")
        
        if self.end_id != -1:
            return infos[:self.end_id]
        else:
            return infos

    def _scan_directory(self):
        """Scan VKitti2 directory to build dataset."""
        infos = []
        
        # Build path to scene/variation
        scene_path = osp.join(self.data_prefix, self.scene, self.variation, 'frames')
        
        if not osp.exists(scene_path):
            print(f"Warning: Scene path {scene_path} does not exist!")
            return infos
        
        # Load intrinsics from VKitti2 intrinsic.txt file
        intrinsic_path = osp.join(self.data_prefix, self.scene, self.variation, 'intrinsic.txt')
        if osp.exists(intrinsic_path):
            _, K_left, K_right = load_vkitti2_odom_intrinsics(
                intrinsic_path, 
                self.target_size[1],  # height
                self.target_size[0]   # width
            )
            focal_length = K_left[0, 0]
        else:
            raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}. "
                                  f"Please ensure VKitti2 dataset is properly structured.")
        
        # Get list of left images
        left_img_dir = osp.join(scene_path, 'rgb', 'Camera_0')
        if not osp.exists(left_img_dir):
            print(f"Warning: Left image directory {left_img_dir} does not exist!")
            return infos
        
        left_images = sorted(glob.glob(osp.join(left_img_dir, '*.jpg')))
        
        if len(left_images) == 0:
            print(f"Warning: No images found in {left_img_dir}")
            return infos
        
        # Create sliding window sequences (e.g., 3-frame sequences)
        window_size = 3
        for i in range(len(left_images) - window_size + 1):
            path = {}
            
            # Get frame indices
            frame_indices = list(range(i, i + window_size))
            
            # Left images
            path["left_frame_paths"] = [
                osp.join(scene_path, 'rgb', 'Camera_0', f'rgb_{idx:05d}.jpg')
                for idx in frame_indices
            ]
            
            # Right images
            path["right_frame_paths"] = [
                osp.join(scene_path, 'rgb', 'Camera_1', f'rgb_{idx:05d}.jpg')
                for idx in frame_indices
            ]
            
            # Use loaded camera intrinsics
            path["k_left"] = K_left
            path["K_right"] = K_right
            path['focal'] = focal_length
            path["focal_right"] = focal_length
            path["baseline"] = self.BASELINE
            path['intrinsics'] = np.stack([K_left, K_right], 0)
            
            # Depth paths if needed
            if self.load_gtdepth:
                path["left_depth_paths"] = [
                    osp.join(scene_path, 'depth', 'Camera_0', f'depth_{idx:05d}.png')
                    for idx in frame_indices
                ]
                path["right_depth_paths"] = [
                    osp.join(scene_path, 'depth', 'Camera_1', f'depth_{idx:05d}.png')
                    for idx in frame_indices
                ]
                path["depth_scale_ratio"] = self.depth_scale_ratio
            
            infos.append(path)
        
        return infos

    def _align_trajectory_sim3(self, pred_xyz, gt_xyz):
        """
        Align predicted trajectory to ground truth using Sim3 alignment 
        (scale + rotation + translation). This is similar to Umeyama alignment.
        
        Args:
            pred_xyz: Predicted trajectory (N, 3)
            gt_xyz: Ground truth trajectory (N, 3)
            
        Returns:
            aligned_pred_xyz: Aligned predicted trajectory (N, 3)
            scale: Scale factor
            R: Rotation matrix (3, 3)
            t: Translation vector (3,)
        """
        # Ensure same length
        min_len = min(len(pred_xyz), len(gt_xyz))
        pred_xyz = pred_xyz[:min_len]
        gt_xyz = gt_xyz[:min_len]
        
        # Compute centroids
        pred_centroid = np.mean(pred_xyz, axis=0)
        gt_centroid = np.mean(gt_xyz, axis=0)
        
        # Center the trajectories
        pred_centered = pred_xyz - pred_centroid
        gt_centered = gt_xyz - gt_centroid
        
        # Compute scale
        pred_scale = np.sqrt(np.mean(np.sum(pred_centered**2, axis=1)))
        gt_scale = np.sqrt(np.mean(np.sum(gt_centered**2, axis=1)))
        scale = gt_scale / pred_scale if pred_scale > 0 else 1.0
        
        # Scale the prediction
        pred_scaled = pred_centered * scale
        
        # Compute rotation using SVD
        H = pred_scaled.T @ gt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply rotation
        pred_rotated = (R @ pred_scaled.T).T
        
        # Compute translation
        t = gt_centroid - pred_centroid * scale
        
        # Final aligned trajectory
        aligned_pred_xyz = pred_rotated + gt_centroid
        
        return aligned_pred_xyz, scale, R, t

    def evaluate(self, results, gt_labels=None, metrics='EPE', logger=None, eval_config=None, **kwargs):
        """
        Evaluate results.
        
        Args:
            results: Prediction results
            gt_labels: Ground truth labels
            metrics: Evaluation metrics
            logger: Logger for output
            eval_config: Evaluation configuration
        """
        
        # Depth evaluation if depth predictions are available
        if len(results) > 0 and isinstance(results[0], list) and len(results[0]) > 0:
            try:
                all_pred_depth = results[0]  # batch views h w 
                all_gt_depth = results[1] if len(results) > 1 else []
                
                if len(all_gt_depth) > 0:
                    min_depth = self.eval_range[0]
                    max_depth = self.eval_range[1]
                    
                    B = len(all_pred_depth)
                    V, H, W = all_pred_depth[0].shape
                    
                    # Compute depth metrics
                    abs_rel = np.zeros(B, np.float32)
                    sq_rel = np.zeros(B, np.float32)
                    rms = np.zeros(B, np.float32)
                    log_rms = np.zeros(B, np.float32)
                    a1 = np.zeros(B, np.float32)
                    a2 = np.zeros(B, np.float32)
                    a3 = np.zeros(B, np.float32)
                    
                    for i in range(B):
                        gt_depth = all_gt_depth[i][0]  # Left view
                        pred_depth = all_pred_depth[i][0]
                        
                        # Clip predictions
                        pred_depth = np.clip(pred_depth, min_depth, max_depth)
                        
                        # Valid mask
                        mask = (gt_depth > min_depth) & (gt_depth < max_depth)
                        
                        if mask.sum() > 0:
                            gt_valid = gt_depth[mask]
                            pred_valid = pred_depth[mask]
                            
                            # Compute metrics
                            thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
                            a1[i] = (thresh < 1.25).mean()
                            a2[i] = (thresh < 1.25 ** 2).mean()
                            a3[i] = (thresh < 1.25 ** 3).mean()
                            
                            abs_rel[i] = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
                            sq_rel[i] = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
                            
                            rms[i] = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
                            log_rms[i] = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
                    
                    # Print results
                    msg = f"\nVKitti2 {self.scene}/{self.variation} Depth Evaluation"
                    print_log(msg, logger=logger)
                    
                    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
                        'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
                    print("{:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f}".format(
                        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), 
                        a1.mean()*100, a2.mean()*100, a3.mean()*100))
                    
                    eval_res = {
                        'abs_rel': abs_rel.mean(), 
                        'sq_rel': sq_rel.mean(), 
                        'rms': rms.mean(), 
                        'log_rms': log_rms.mean(), 
                        'a1': a1.mean()*100, 
                        'a2': a2.mean()*100, 
                        'a3': a3.mean()*100
                    }
                    
                    log_msg = [f'\nVKitti2 depth evaluation results:']
                    for k, acc in eval_res.items():
                        log_msg.append(f'  {k}: {acc:.4f}  ')
                    log_msg = ' '.join(log_msg)
                    print_log(log_msg, logger=logger)
                    
                    return eval_res
            except Exception as e:
                print(f"VKitti2 Depth evaluation error: {e}")
        
        # Pose evaluation if pose predictions are available
        if len(results) > 4 and len(results[4]) > 0:
            print_log("\n=== VKitti2 Pose Evaluation ===", logger=logger)
            if eval_config is not None:  # have prediction of pose
                # Save predicted poses in KITTI format
                print_log("Saving predicted poses in KITTI format...", logger=logger)
                pred_relative_pose_list = results[4].copy()
                pred_abs_pose_list = pose_relative2absolute(pred_relative_pose_list)
                time_str = '_'.join(time.asctime(time.localtime()).split(' '))
                time_str = '_'.join(time_str.split(':'))
                result_dir = osp.join(eval_config["cfg"].work_dir, 
                                        f"pred_poses_{self.test_seq_id}_" + time_str + str(random.randrange(10000, 19999)))
                pred_pose_path = osp.join(result_dir, f"{self.test_seq_id}.txt")
                if not osp.exists(result_dir):
                    os.makedirs(result_dir)
                print_log(f"Saving {len(pred_abs_pose_list)} predicted poses to {pred_pose_path}", logger=logger)
                with open(pred_pose_path, 'w') as f:
                    for p_idx in range(len(pred_abs_pose_list)):
                        str_pose = [str(pp) for pp in pred_abs_pose_list[p_idx].reshape(-1)[:12].tolist()]
                        f.write(' '.join(str_pose) + '\n')
                
                # save gt pose into .txt with kitti format
                if len(results) > 5 and len(results[5]) > 0:
                    gt_relative_pose_list = results[5].copy()
                    gt_abs_pose_list = pose_relative2absolute(gt_relative_pose_list)
                    gt_dir = osp.join(eval_config["cfg"].work_dir, "gt_poses")
                    if not osp.exists(gt_dir):
                        os.makedirs(gt_dir)
                    gt_pose_path = osp.join(gt_dir, f"{self.test_seq_id}.txt")
                    with open(gt_pose_path, 'w') as f:
                        for p_idx in range(len(gt_abs_pose_list)):
                            str_pose = [str(pp) for pp in gt_abs_pose_list[p_idx].reshape(-1)[:12].tolist()]
                            f.write(' '.join(str_pose) + '\n')
                    
                    print_log(f"Saved {len(gt_abs_pose_list)} ground truth poses to {gt_pose_path}", logger=logger)
                    
                    # Import evaluation tools
                    try:
                        from kitti_odom_eval.kitti_odometry import KittiEvalOdom
                        from KITTI_odometry_evaluation_tool.evaluation import kittiOdomEval
                        
                        # Both tools now support string sequence IDs directly
                        print(f"\n{'='*70}")
                        print(f"=== ORIGINAL Pose Evaluation for {self.test_seq_id} ===")
                        print(f"{'='*70}")
                        
                        # Evaluation tool 1: KittiEvalOdom
                        print(f"\n--- Evaluation Tool 1: KittiEvalOdom (Original) ---")
                        eval_tool = KittiEvalOdom(dataset_type='kitti')
                        eval_tool.eval(
                            gt_dir,
                            result_dir,
                            alignment='7dof',  # ['scale', 'scale_7dof', '7dof', '6dof']
                            seqs=[self.test_seq_id],
                            plot_keys=[str(self.test_seq_id) + "_original_" + str(time.time())]
                        )
                        
                        # Evaluation tool 2: kittiOdomEva
                        print(f"\n--- Evaluation Tool 2: kittiOdomEval (Original) ---")
                        dict_tool2 = {
                            "gt_dir": gt_dir, 
                            "result_dir": result_dir, 
                            "eva_seqs": str(self.test_seq_id),
                            "toCameraCoord": False
                        }
                        pose_eval = kittiOdomEval(dict_tool2)
                        pose_eval.eval(toCameraCoord=dict_tool2['toCameraCoord'])
                        
                        print(f"\n{'='*70}")
                        print(f"=== All Pose Evaluations Completed for {self.test_seq_id} ===")
                        print(f"{'='*70}\n")
                    except Exception as e:
                        import traceback
                        print(f"VKitti2 Pose evaluation error: {e}")
                        print(traceback.format_exc())
                        print("Please check if kitti_odom_eval and KITTI_odometry_evaluation_tool are properly installed.")
   
        
        return {}
