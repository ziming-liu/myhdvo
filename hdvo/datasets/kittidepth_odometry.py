'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 16:42:40
LastEditors: Ziming Liu
LastEditTime: 2024-04-22 00:12:45
'''
from ftplib import all_errors
import os.path as osp
from turtle import right
from typing import Sequence
import numpy as np
from mmcv.utils import print_log
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
 
from hdvo.models.utils.pose_utils import pose_relative2absolute
from kitti_odom_eval.kitti_odometry import KittiEvalOdom
from KITTI_odometry_evaluation_tool.evaluation import kittiOdomEval

@DATASETS.register_module()
class KITTIOdometryDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, depth_scale_ratio=256, data_prefix=None, test_mode=False, end_id=-1,
                 eval_modality='disparity', eval_range=[1,192], filename_tmpl='{:0>10}.png',
                   d_filename_tmpl='{:0>10}.png', crop_test_image='garg', camera="23",
                     test_seq_id=99, load_gtdepth=False, kitti_rawdata_path=None, **kwargs):
        super().__init__(ann_file=ann_file,
                 pipeline=pipeline,
                 data_prefix=data_prefix,
                 depth_scale_ratio=depth_scale_ratio,
                 test_mode=test_mode,
                 eval_modality=eval_modality,
                 eval_range=eval_range,
                 filename_tmpl=filename_tmpl, 
                 d_filename_tmpl=d_filename_tmpl, )
        self.kitti_rawdata_path = kitti_rawdata_path
        self.test_seq_id = test_seq_id
        self.camera = camera
        self.pip = pipeline
        self.end_id = end_id
        self.test_mode = test_mode
        self.crop_test_image = crop_test_image
        self.load_gtdepth = load_gtdepth
        self.video_infos = self.load_annotations()
        print("num samples: ", len(self.video_infos))


    def load_annotations(self):
        rawdata = mmcv.load(self.ann_file)
        num_ = len(rawdata)
        infos = []
        for i in range(num_):
            path = {}
            if self.load_gtdepth and "/03/" in rawdata[i]["image_2_paths"][0]: # 03 seq missed gtdepth
                continue
            if self.camera=="01":
                path["left_frame_paths"] = [os.path.join(self.data_prefix, *a.split('/')[7:]) for a in rawdata[i]["image_0_paths"]]
                path["right_frame_paths"] = [os.path.join(self.data_prefix, *a.split('/')[7:]) for a in rawdata[i]["image_1_paths"]]
                path["k_left"] = np.array([float(i) for i in rawdata[i]["K_0"].strip().split(' ')]).reshape(3,4)[:3,:3].astype(np.float32)
                path["K_right"] = np.array([float(i) for i in rawdata[i]["K_1"].strip().split(' ')]).reshape(3,4)[:3,:3].astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_0"])
                path["focal_right"] = float(rawdata[i]["focal_1"])
                path["baseline"] = float(rawdata[i]["baseline_01"])
            elif self.camera=="23":
                path["left_frame_paths"] = [os.path.join(self.data_prefix, *a.split('/')[7:]) for a in rawdata[i]["image_2_paths"]] 
                path["right_frame_paths"] = [os.path.join(self.data_prefix, *a.split('/')[7:]) for a in rawdata[i]["image_3_paths"]]
                path["k_left"] = np.array([float(i) for i in rawdata[i]["K_2"].strip().split(' ')]).reshape(3,4)[:3,:3].astype(np.float32)
                path["K_right"] = np.array([float(i) for i in rawdata[i]["K_3"].strip().split(' ')]).reshape(3,4)[:3,:3].astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_2"])
                path["focal_right"] = float(rawdata[i]["focal_3"])
                path["baseline"] = float(rawdata[i]["baseline_23"])
                if self.load_gtdepth:
                    path["left_depth_paths"] = [path["left_frame_paths"][i].replace("image_2", "depth_2") for i in range(len(path["left_frame_paths"]))]
                    path["right_depth_paths"] = [path["right_frame_paths"][i].replace("image_3", "depth_3") for i in range(len(path["right_frame_paths"]))]
                    path["depth_scale_ratio"] = self.depth_scale_ratio
            assert (path["k_left"] == path["K_right"]).all()
            assert (path['focal'] == path["focal_right"] )
            path['intrinsics'] = np.stack([path["k_left"], path["K_right"]],0)
            if "gt_poses" in rawdata[0] and rawdata[0]["gt_poses"] is not None:
                pose = [np.array([float(i) for i in a.strip().split(' ')]).reshape(3,4) for a in rawdata[i]["gt_poses"]]
                pose = [ np.concatenate([a, np.array([[0,0,0,1]])], 0) for a in pose]
                path["pose"] = np.stack(pose).astype(np.float32)
            
            self.seq_dir = '/'.join(path["left_frame_paths"][0].split("/")[:-2]) 
            
            infos.append(path)
        if self.end_id !=-1:
            return  infos[:self.end_id]
        else:
            return infos
    
    def load_annotations_test(self):
        rawdata = mmcv.load(self.ann_file)
        num_ = len(rawdata)
        infos = []
        for i in range(num_):
            path = {}
            if "gt_poses" in rawdata[i]:
                path["pose"] = np.stack([np.array([float(i) for i in a.strip().split(' ')]).reshape(4,4) for a in rawdata[i]["gt_poses"]]).astype(np.float32)
            if self.camera=="01":
                path["left_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_0_paths"]]
                path["right_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_1_paths"]]
                path["k_left"] = np.array([float(i) for i in rawdata[i]["K_0"].strip().split(' ')]).reshape(3,3).astype(np.float32)
                path["K_right"] = np.array([float(i) for i in rawdata[i]["K_1"].strip().split(' ')]).reshape(3,3).astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_0"])
                path["focal_right"] = float(rawdata[i]["focal_1"])
                path["baseline"] = float(rawdata[i]["baseline_01"])
            elif self.camera=="23":
                path["left_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_2_paths"]] 
                path["right_frame_paths"] = [os.path.join(self.data_prefix, a) for a in rawdata[i]["image_3_paths"]]
                path["k_left"] = np.array([float(i) for i in rawdata[i]["K_2"].strip().split(' ')]).reshape(3,3).astype(np.float32)
                path["K_right"] = np.array([float(i) for i in rawdata[i]["K_3"].strip().split(' ')]).reshape(3,3).astype(np.float32)
                path['focal'] = float(rawdata[i]["focal_2"])
                path["focal_right"] = float(rawdata[i]["focal_3"])
                path["baseline"] = float(rawdata[i]["baseline_23"])
            assert (path["k_left"] == path["K_right"]).all()
            assert (path['focal'] == path["focal_right"] )
            path['intrinsics'] = np.stack([path["k_left"], path["K_right"]],0)
            infos.append(path)
        print(f"num samples {len(infos[:self.end_id])}")
        return  infos[:self.end_id]
    
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
        
        gt_path = self.kitti_rawdata_path
        assert gt_path is not None, "you have to provide kitti_rawdata_path for odometry depth evaluation"
        #print("### start to evaluate KITTI DEPTH on odometry seq {} ###".format(self.scene_list))
        min_depth = 1#self.depth_range[0] #2.018
        max_depth = 80 #self.depth_range[1]
        garg_crop = True
        eigen_crop = False
        test_kbcrop= False
        
        #garg_crop = True if self.test_edgecrop=='garg_crop' else False
        #eigen_crop = True if self.test_edgecrop =='eigen_crop' else False
        masked_percent_depth_range = 0
        masked_percent_crop = 0
        masked_percent_multimask = 0
        masked_percent_total = 0

        try:
            if int(self.test_seq_id)<=10:
                all_pred_depth = results[0] # batch views h w 
                all_gt_depth = results[1] # batch views h w 
                all_multimasks = results[2]
                with_mask = True
                if len(all_multimasks) == 0: with_mask = False
                B = len(all_pred_depth)
                V, H, W = all_pred_depth[0].shape
                total_pixels=  H*W
                if V==1: views = ["left"]
                elif V==2: views = ["left", "right"]
                else: raise ValueError

                view = views[0]
                if len(all_gt_depth) > 0:
                    gt_depths = [np.squeeze(all_gt_depth[k][0, :, :]) for k in range(B)]
                    pred_depths = [np.squeeze(all_pred_depth[k][0, :, :]) for k in range(B)]
                    num_samples = len(gt_depths)
                else:
                    for idx, view in enumerate(views):
                        pred_disparities = [all_pred_depth[k][idx, :, :] for k in range(B)]
                        multimask = [all_multimasks[k][idx, :, :] for k in range(B)] if with_mask == True else None
                    
                        #num_samples = 697
                        #test_files = read_text_lines('/data/acentauri/user/ziliu/data/depth_splits/eigen_test_files.txt')
                        # for test stage, there is only one sequence id in self.seq_dir
                        test_files = read_text_lines(os.path.join(self.seq_dir, "gt_depth_ann_files.txt"))
                        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path, view)
                        
                        #num_test = len(im_files)
                        gt_depths = []
                        pred_depths = []
                        print("# load gt depth")
                        assert len(gt_files) >= len(pred_disparities), "len of GT files: {},  len of pred disps: {}".format(len(gt_files),len(pred_disparities))
                        num_samples = len(pred_disparities)
                        for t_id in tqdm(range(num_samples)):
                            camera_id = cams[t_id]  # 2 is left, 3 is right
                            #print("gt calib {}".format(gt_calib[t_id]))
                            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
                            depth = depth.astype(np.float32)
                            #if not os.path.exists(os.path.join(self.seq_dir, f"depth_{camera_id}")):
                            #    os.makedirs(os.path.join(self.seq_dir, f"depth_{camera_id}"))
                            #    cv2.imwrite(os.path.join(self.seq_dir, f"depth_{camera_id}", "{:0>10}.png".format(t_id)), (256*depth).astype(np.uint16))
                            # kb cropping
                            def cropping(img):
                                h_im, w_im = img.shape[:2]
                                self.margin_top = int(h_im - 352)
                                self.margin_left = int((w_im - 1216) / 2)

                                img = img[self.margin_top: self.margin_top + 352,
                                        self.margin_left: self.margin_left + 1216]
                                return img
                            #if depth.shape !=pred_disparities[t_id].shape:
                            if test_kbcrop:
                                depth = cropping(depth)
                            #if self.test_edgecrop is None:
                            #    h_img, w_img = depth.shape[-2:]
                            #    crop = np.array([0.10810811 * h_img //16*16,  0.99189189 * h_img//16*16,   
                            #                        0.03594771 * w_img //16*16,   0.96405229 * w_img//16*16]).astype(np.int32)
                            #    depth = depth[crop[0]:crop[1], crop[2]:crop[3]]
                            gt_depths.append(depth)
                            # im_sizes: (h, w)
                            #disp_pred = pred_disparities[t_id]
                            #assert disp_pred.shape== depth.shape
                            if pred_disparities[t_id].shape[-2:]!= depth.shape[-2:]:
                                disp_pred = cv2.resize(pred_disparities[t_id], (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_LINEAR)
                            else:
                                disp_pred = pred_disparities[t_id]
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

                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

                    
                    if garg_crop or eigen_crop:
                        gt_height, gt_width = gt_depth.shape

                        # crop used by Garg ECCV16
                        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                        if garg_crop:
                            #crop = np.array([0.40810811 * gt_height,  0.8 * gt_height,   
                            #                0.3594771 * gt_width,   0.7* gt_width]).astype(np.int32)
                            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                            0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                        # crop we found by trial and error to reproduce Eigen NIPS14 results
                        elif eigen_crop:
                            crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                            0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                        crop_mask = np.zeros(mask.shape)
                        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                        mask = np.logical_and(mask, crop_mask)
                    abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
                
                msg = "seq {} view {} Evaluating ...".format(self.seq_dir, view)
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
                print("{:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean()*100, a2.mean()*100, a3.mean()*100))
                eval_res = {'abs_rel':abs_rel.mean(), 'sq_rel': sq_rel.mean(), 'rms':rms.mean(), 'log_rms': log_rms.mean(), 'd1_all':d1_all.mean(), 'a1':a1.mean()*100, 'a2':a2.mean()*100, 'a3': a3.mean()*100}
                log_msg = []
                log_msg.append(f'\n evaluation results')
                for k, acc in eval_res.items():
                    #depth_eval_results[f'_{k}'] = acc
                    log_msg.append(f'  {k}: {acc:.4f}  ')
                    #self.logger.info(f'  {k}: {acc:.4f}  ')
                log_msg = ' '.join(log_msg)
                print_log(log_msg, logger=logger)
        except:
            print("KITTI Depth evaluation error!")
            pass
        
        print("### KITTI ODOMETRY POSE EVALUATION ###")
        print(results[4])
        if len(results[4]) >0 and eval_config is not None: # have predition of pose
            # save pred pose into .txt with kitti format
            print("save pred pose into .txt with kitti format")
            pred_relative_pose_list = results[4].copy()
            pred_abs_pose_list = pose_relative2absolute(pred_relative_pose_list)
            time_str = '_'.join( time.asctime(time.localtime()).split(' ') )
            time_str = '_'.join( time_str.split(':') )
            result_dir = os.path.join(eval_config["cfg"].work_dir, f"pred_poses_{self.test_seq_id}_"+time_str+str(random.randrange(10000,19999)))
            pred_pose_path = os.path.join(result_dir, "{}.txt".format(self.test_seq_id))
            print(f"save pred pose into {pred_pose_path}")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            print(f"num of pose {len(pred_abs_pose_list)}")
            with open(pred_pose_path, 'w') as f:
                for p_idx in range(len(pred_abs_pose_list)):
                    str_pose =  [  str(pp) for pp in pred_abs_pose_list[p_idx].reshape(-1)[:12].tolist()]
                    f.write(' '.join(str_pose)+'\n')
            # save gt pose into .txt with kitti format
            gt_relative_pose_list = results[5].copy()
            gt_abs_pose_list = pose_relative2absolute(gt_relative_pose_list)
            gt_pose_path = os.path.join(eval_config["cfg"].work_dir, "gt_poses", "{}.txt".format(self.test_seq_id))
            gt_dir = os.path.join(eval_config["cfg"].work_dir, "gt_poses")
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            with open(gt_pose_path, 'w') as f:
                for p_idx in range(len(gt_abs_pose_list)):
                    str_pose =  [  str(pp) for pp in gt_abs_pose_list[p_idx].reshape(-1)[:12].tolist()]
                    f.write(' '.join(str_pose)+'\n')
            
            # Apply SIM3 alignment and save aligned poses
            print("\n### Applying SIM3 alignment to predicted poses ###")
            pred_xyz = np.array([pose[:3, 3] for pose in pred_abs_pose_list])
            gt_xyz = np.array([pose[:3, 3] for pose in gt_abs_pose_list])
            
            aligned_pred_xyz, scale, R, t = self._align_trajectory_sim3(pred_xyz, gt_xyz)
            print(f"SIM3 alignment - Scale factor: {scale:.6f}")
            
            # Create aligned pose matrices
            pred_abs_pose_list_aligned = []
            for p_idx in range(len(pred_abs_pose_list)):
                aligned_pose = pred_abs_pose_list[p_idx].copy()
                # Apply rotation and scale to translation
                aligned_pose[:3, 3] = aligned_pred_xyz[p_idx]
                # Apply rotation to rotation part
                aligned_pose[:3, :3] = R @ aligned_pose[:3, :3]
                pred_abs_pose_list_aligned.append(aligned_pose)
            
            # Save aligned poses
            result_dir_aligned = os.path.join(eval_config["cfg"].work_dir, 
                                                f"pred_poses_{self.test_seq_id}_aligned_sim3_"+time_str+str(random.randrange(10000,19999)))
            if not os.path.exists(result_dir_aligned):
                os.makedirs(result_dir_aligned)
            pred_pose_path_aligned = os.path.join(result_dir_aligned, "{}.txt".format(self.test_seq_id))
            
            with open(pred_pose_path_aligned, 'w') as f:
                for p_idx in range(len(pred_abs_pose_list_aligned)):
                    str_pose =  [  str(pp) for pp in pred_abs_pose_list_aligned[p_idx].reshape(-1)[:12].tolist()]
                    f.write(' '.join(str_pose)+'\n')
            
            print(f"Saved SIM3-aligned poses to {pred_pose_path_aligned}")
            print(f"Number of aligned poses: {len(pred_abs_pose_list_aligned)}")
            
            # evaluate pose estimation 
            print(self.test_seq_id)
            if int(self.test_seq_id) >10: return 0
            
            print("\n" + "="*70)
            print(f"=== ORIGINAL Pose Evaluation for {self.test_seq_id} ===")
            print("="*70)
            eval_tool = KittiEvalOdom()
            
            
            print("Evaluate result in  {}".format(result_dir))
            

            if eval_config["cfg"].dataset_type != "EurocMavDataset" and eval_config["cfg"].dataset_type != "MidAirDataset" :
                eval_tool.eval(
                        gt_dir,
                        result_dir,
                        alignment=None, # ['scale', 'scale_7dof', '7dof', '6dof'],
                        seqs=[str(self.test_seq_id)], # e.g. 09,09ep2,09ep3,09ep4 10,10ep2",
                        plot_keys=[str(self.test_seq_id)+"_original_"+str(time.time())] # + args.checkpoint.split('.')[-2].split('/')[-1]  #  e.g. 10epoch1 10epoch2 ",
                        )
            
            # eval tool2
            dict_tool2 = {"gt_dir":gt_dir, "result_dir": result_dir, "eva_seqs": f"{self.test_seq_id}_pred",\
                "toCameraCoord": False}
            print(dict_tool2)
            pose_eval = kittiOdomEval(dict_tool2)
            pose_eval.eval(toCameraCoord=dict_tool2['toCameraCoord'])   # set the value according to the predicted results
            
            # Evaluate aligned poses
            print("\n" + "="*70)
            print(f"=== SIM3-ALIGNED Pose Evaluation for {self.test_seq_id} ===")
            print("="*70)
            
            eval_tool_aligned = KittiEvalOdom()
            print("Evaluate aligned result in  {}".format(result_dir_aligned))
            
            if eval_config["cfg"].dataset_type != "EurocMavDataset" and eval_config["cfg"].dataset_type != "MidAirDataset" :
                eval_tool_aligned.eval(
                        gt_dir,
                        result_dir_aligned,
                        alignment=None, # Already aligned, so no additional alignment
                        seqs=[str(self.test_seq_id)],
                        plot_keys=[str(self.test_seq_id)+"_sim3_aligned_"+str(time.time())]
                        )
            
            # eval tool2 for aligned poses
            dict_tool2_aligned = {"gt_dir":gt_dir, "result_dir": result_dir_aligned, "eva_seqs": f"{self.test_seq_id}_pred",\
                "toCameraCoord": False}
            print(dict_tool2_aligned)
            pose_eval_aligned = kittiOdomEval(dict_tool2_aligned)
            pose_eval_aligned.eval(toCameraCoord=dict_tool2_aligned['toCameraCoord'])
            
            print("\n" + "="*70)
            print(f"=== All Pose Evaluations Completed for {self.test_seq_id} ===")
            print("="*70 + "\n")




