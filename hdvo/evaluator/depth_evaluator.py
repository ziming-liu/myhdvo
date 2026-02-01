"""
Depth Evaluator Module.

This module provides an independent depth evaluator for KITTI Odometry dataset,
decoupled from dataset classes.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from mmcv.utils import print_log
import warnings

from .evaluation_utils import (
    read_text_lines, 
    read_file_data, 
    generate_depth_map, 
    compute_errors
)


class DepthEvaluator:
    """Evaluator for depth estimation on KITTI Odometry dataset.
    
    Args:
        min_depth (float): Minimum depth threshold. Default: 1.0.
        max_depth (float): Maximum depth threshold. Default: 80.0.
        garg_crop (bool): Whether to use Garg ECCV16 crop. Default: True.
        eigen_crop (bool): Whether to use Eigen NIPS14 crop. Default: False.
        test_kbcrop (bool): Whether to use KB crop. Default: False.
    """
    
    def __init__(self, min_depth=1.0, max_depth=80.0, garg_crop=True, 
                 eigen_crop=False, test_kbcrop=False):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.garg_crop = garg_crop
        self.eigen_crop = eigen_crop
        self.test_kbcrop = test_kbcrop
    
    def evaluate(self, all_pred_depth, all_gt_depth=None, all_multimasks=None, 
                 seq_dir=None, gt_path=None, logger=None):
        """Main evaluation interface.
        
        Args:
            all_pred_depth (list): List of predicted depth maps.
            all_gt_depth (list, optional): List of ground truth depth maps. 
                If provided, evaluate directly. Otherwise use raw data.
            all_multimasks (list, optional): List of masks.
            seq_dir (str, optional): Sequence directory path.
            gt_path (str, optional): Path to KITTI raw data (for raw data evaluation).
            logger: Logger for output.
            
        Returns:
            dict: Evaluation metrics.
        """
        if all_multimasks is None:
            all_multimasks = []
        
        # Determine evaluation mode
        if all_gt_depth is not None and len(all_gt_depth) > 0:
            return self.evaluate_with_gt_depth(
                all_pred_depth, all_gt_depth, all_multimasks, seq_dir, logger
            )
        else:
            if gt_path is None:
                raise ValueError("gt_path must be provided when all_gt_depth is None")
            return self.evaluate_with_raw_data(
                all_pred_depth, all_multimasks, seq_dir, gt_path, logger
            )
    
    def evaluate_with_gt_depth(self, all_pred_depth, all_gt_depth, all_multimasks, 
                               seq_dir, logger=None):
        """Evaluate depth when ground truth depth is directly available.
        
        Args:
            all_pred_depth (list): List of predicted depth maps.
            all_gt_depth (list): List of ground truth depth maps.
            all_multimasks (list): List of masks.
            seq_dir (str): Sequence directory path.
            logger: Logger for output.
            
        Returns:
            dict: Evaluation metrics.
        """
        with_mask = len(all_multimasks) > 0
        B = len(all_pred_depth)
        V, H, W = all_pred_depth[0].shape
        
        if V == 1:
            views = ["left"]
        elif V == 2:
            views = ["left", "right"]
        else:
            raise ValueError(f"Invalid number of views: {V}")

        view = views[0]
        gt_depths = [np.squeeze(all_gt_depth[k][0, :, :]) for k in range(B)]
        pred_depths = [np.squeeze(all_pred_depth[k][0, :, :]) for k in range(B)]
        num_samples = len(gt_depths)
        
        return self._compute_depth_metrics(pred_depths, gt_depths, num_samples, 
                                          seq_dir, view, logger)
    
    def evaluate_with_raw_data(self, all_pred_depth, all_multimasks, seq_dir, 
                               gt_path, logger=None):
        """Evaluate depth using KITTI raw data for ground truth generation.
        
        Args:
            all_pred_depth (list): List of predicted depth maps.
            all_multimasks (list): List of masks.
            seq_dir (str): Sequence directory path.
            gt_path (str): Path to KITTI raw data.
            logger: Logger for output.
            
        Returns:
            dict: Evaluation metrics.
        """
        with_mask = len(all_multimasks) > 0
        B = len(all_pred_depth)
        V, H, W = all_pred_depth[0].shape
        
        if V == 1:
            views = ["left"]
        elif V == 2:
            views = ["left", "right"]
        else:
            raise ValueError(f"Invalid number of views: {V}")

        eval_results = {}
        
        for idx, view in enumerate(views):
            pred_disparities = [all_pred_depth[k][idx, :, :] for k in range(B)]
            multimask = [all_multimasks[k][idx, :, :] for k in range(B)] if with_mask else None

            if not os.path.exists(os.path.join(seq_dir, "gt_depth_ann_files.txt")):
                warnings.warn(f"Sequence directory {seq_dir} does not exist. Skipping evaluation for {view} view.")
                continue
            test_files = read_text_lines(os.path.join(seq_dir, "gt_depth_ann_files.txt"))
            gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path, view)
            
            gt_depths = []
            pred_depths = []
            print(f"Loading ground truth depth for {view} view...")
            assert len(gt_files) >= len(pred_disparities), \
                f"GT files ({len(gt_files)}) < predicted disparities ({len(pred_disparities)})"
            num_samples = len(pred_disparities)
            
            for t_id in tqdm(range(num_samples), desc=f"Processing {view} depth"):
                camera_id = cams[t_id]
                depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], 
                                         im_sizes[t_id], camera_id, False, True)
                depth = depth.astype(np.float32)
                
                # Apply cropping if needed
                if self.test_kbcrop:
                    h_im, w_im = depth.shape[:2]
                    margin_top = int(h_im - 352)
                    margin_left = int((w_im - 1216) / 2)
                    depth = depth[margin_top:margin_top + 352, 
                                margin_left:margin_left + 1216]
                
                gt_depths.append(depth)
                
                # Resize prediction if needed
                if pred_disparities[t_id].shape[-2:] != depth.shape[-2:]:
                    disp_pred = cv2.resize(pred_disparities[t_id], 
                                         (depth.shape[-1], depth.shape[-2]), 
                                         interpolation=cv2.INTER_LINEAR)
                else:
                    disp_pred = pred_disparities[t_id]
                
                depth_pred = disp_pred
                depth_pred[np.isinf(depth_pred)] = 0
                pred_depths.append(depth_pred)
            
            result = self._compute_depth_metrics(pred_depths, gt_depths, num_samples, 
                                                 seq_dir, view, logger)
            eval_results[view] = result
        
        return eval_results.get('left', eval_results)
    
    def _compute_depth_metrics(self, pred_depths, gt_depths, num_samples, 
                               seq_dir, view, logger=None):
        """Compute depth evaluation metrics.
        
        Args:
            pred_depths (list): List of predicted depth maps.
            gt_depths (list): List of ground truth depth maps.
            num_samples (int): Number of samples.
            seq_dir (str): Sequence directory path.
            view (str): Camera view name.
            logger: Logger for output.
            
        Returns:
            dict: Evaluation metrics.
        """
        pred_depths = pred_depths[:num_samples]
        assert len(pred_depths) == len(gt_depths), \
            f"pred_depths ({len(pred_depths)}) != gt_depths ({len(gt_depths)})"
        
        # Initialize metrics
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1_all = np.zeros(num_samples, np.float32)
        a1 = np.zeros(num_samples, np.float32)
        a2 = np.zeros(num_samples, np.float32)
        a3 = np.zeros(num_samples, np.float32)
        
        print("Computing depth metrics...")
        for i in tqdm(range(num_samples), desc="Computing metrics"):
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]
            pred_depth[pred_depth < self.min_depth] = self.min_depth
            pred_depth[pred_depth > self.max_depth] = self.max_depth

            mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)

            if self.garg_crop or self.eigen_crop:
                gt_height, gt_width = gt_depth.shape

                if self.garg_crop:
                    # Crop used by Garg ECCV16
                    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,   
                                   0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                elif self.eigen_crop:
                    # Crop used by Eigen NIPS14
                    crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,   
                                   0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
                
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
                compute_errors(gt_depth[mask], pred_depth[mask])
        
        msg = f"Sequence {seq_dir} view {view} - Evaluating depth..."
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
        print("{:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f} & {:10.4f}".format(
            abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), 
            d1_all.mean(), a1.mean()*100, a2.mean()*100, a3.mean()*100))
        
        eval_res = {
            'abs_rel': abs_rel.mean(), 
            'sq_rel': sq_rel.mean(), 
            'rms': rms.mean(), 
            'log_rms': log_rms.mean(), 
            'd1_all': d1_all.mean(), 
            'a1': a1.mean()*100, 
            'a2': a2.mean()*100, 
            'a3': a3.mean()*100
        }
        
        log_msg = '\nDepth evaluation results: '
        log_msg += '  '.join([f'{k}: {acc:.4f}' for k, acc in eval_res.items()])
        print_log(log_msg, logger=logger)
        
        return eval_res
