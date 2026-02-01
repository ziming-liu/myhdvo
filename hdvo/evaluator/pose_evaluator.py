"""
Integrated Pose Evaluator Module.

This module provides a self-contained pose evaluator for KITTI odometry evaluation.
All evaluation functionality is integrated directly without external dependencies on
kitti_odom_eval and KITTI_odometry_evaluation_tool.
"""

import os
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import matplotlib.backends.backend_pdf
from mmcv.utils import print_log

from hdvo.models.utils.pose_utils import pose_relative2absolute


class PoseEvaluator:
    """Integrated evaluator for pose estimation on KITTI Odometry dataset.
    
    This evaluator provides all necessary evaluation metrics and visualizations:
    - Relative pose error (translation & rotation) over different trajectory lengths
    - Absolute trajectory error (ATE) 
    - Relative pose error (RPE)
    - SIM3 alignment (scale + rotation + translation)
    - Trajectory visualization (2D, 3D, XYZ, RPY plots)
    - Error plots per segment length
    
    Args:
        work_dir (str): Working directory for saving results.
        test_seq_id (str or int): Test sequence ID.
        dataset_type (str): Dataset type. Default: 'KITTIOdometryDataset'.
        apply_sim3_alignment (bool): Whether to apply SIM3 alignment. Default: True.
        gt_pose_dir (str, optional): Ground truth pose directory. If None, uses work_dir/gt_poses.
        lengths (list, optional): List of trajectory lengths for evaluation. Default: [100, 200, 300, 400, 500, 600, 700, 800].
        step_size (int, optional): Frame step size for evaluation. Default: 10 (10Hz for KITTI).
    """
    
    def __init__(self, work_dir, test_seq_id, dataset_type='KITTIOdometryDataset',
                 apply_sim3_alignment=True, gt_pose_dir=None,
                 lengths=None, step_size=10):
        self.work_dir = work_dir
        self.test_seq_id = str(test_seq_id)
        self.dataset_type = dataset_type
        self.apply_sim3_alignment = apply_sim3_alignment
        self.gt_pose_dir = gt_pose_dir or os.path.join(work_dir, "gt_poses")
        
        # Evaluation parameters
        self.lengths = lengths if lengths is not None else [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
        self.step_size = step_size
        
    def evaluate(self, pred_relative_poses, gt_relative_poses=None, logger=None):
        """Evaluate pose estimation results.
        
        Args:
            pred_relative_poses (list or np.ndarray): List of predicted relative poses (4x4 matrices).
            gt_relative_poses (list or np.ndarray, optional): List of ground truth relative poses.
                If None, ground truth will be loaded from gt_pose_dir.
            logger: Logger for output.
            
        Returns:
            dict: Evaluation results containing:
                - 'original': Results from original poses
                - 'sim3_aligned': Results from SIM3-aligned poses (if enabled)
        """
        print_log("\n" + "="*80, logger=logger)
        print_log("=== INTEGRATED KITTI ODOMETRY POSE EVALUATION ===", logger=logger)
        print_log("="*80, logger=logger)
        
        # Convert relative poses to absolute poses
        pred_abs_pose_list = pose_relative2absolute(pred_relative_poses.copy())
        
        if gt_relative_poses is not None:
            gt_abs_pose_list = pose_relative2absolute(gt_relative_poses.copy())
            # Save ground truth poses
            self._save_poses(gt_abs_pose_list, self.gt_pose_dir, self.test_seq_id)
        else:
            # Load ground truth from file
            gt_abs_pose_list = self._load_poses(self.gt_pose_dir, self.test_seq_id)
            if gt_abs_pose_list is None:
                raise ValueError(f"Ground truth poses not found in {self.gt_pose_dir}")
        
        # Generate unique timestamp for result directory
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        random_suffix = random.randrange(10000, 19999)
        
        results = {}
        
        # ========================================
        # Evaluate Original Poses
        # ========================================
        result_dir = os.path.join(
            self.work_dir, 
            f"pred_poses_{self.test_seq_id}_{time_str}_{random_suffix}"
        )
        
        self._save_poses(pred_abs_pose_list, result_dir, self.test_seq_id)
        
        if int(self.test_seq_id) <= 10:
            print_log("\n" + "="*70, logger=logger)
            print_log(f"=== Original Pose Evaluation (Sequence {self.test_seq_id}) ===", logger=logger)
            print_log("="*70, logger=logger)
            
            results['original'] = self._run_evaluation(
                gt_abs_pose_list, pred_abs_pose_list, result_dir,
                plot_key=f"{self.test_seq_id}_original_{time_str}",
                logger=logger
            )
            
            self._print_summary(results['original'], "Original", logger)
        
        # ========================================
        # Evaluate SIM3-Aligned Poses
        # ========================================
        if self.apply_sim3_alignment:
            pred_abs_pose_list_aligned, scale = self._align_poses_sim3(
                pred_abs_pose_list, gt_abs_pose_list
            )
            
            print_log(f"\nSIM3 Alignment - Scale factor: {scale:.6f}", logger=logger)
            
            result_dir_aligned = os.path.join(
                self.work_dir, 
                f"pred_poses_{self.test_seq_id}_sim3_aligned_{time_str}_{random_suffix}"
            )
            
            self._save_poses(pred_abs_pose_list_aligned, result_dir_aligned, self.test_seq_id)
            
            if int(self.test_seq_id) <= 10:
                print_log("\n" + "="*70, logger=logger)
                print_log(f"=== SIM3-Aligned Pose Evaluation (Sequence {self.test_seq_id}) ===", logger=logger)
                print_log("="*70, logger=logger)
                
                results['sim3_aligned'] = self._run_evaluation(
                    gt_abs_pose_list, pred_abs_pose_list_aligned, result_dir_aligned,
                    plot_key=f"{self.test_seq_id}_sim3_{time_str}",
                    logger=logger
                )
                
                self._print_summary(results['sim3_aligned'], "SIM3-Aligned", logger)
        
        print_log("\n" + "="*80, logger=logger)
        print_log(f"=== All Evaluations Completed for Sequence {self.test_seq_id} ===", logger=logger)
        print_log("="*80 + "\n", logger=logger)
        
        return results
    
    def _run_evaluation(self, gt_poses, pred_poses, result_dir, plot_key=None, logger=None):
        """Run complete evaluation with all metrics and visualizations.
        
        Args:
            gt_poses (list): Ground truth pose list (4x4 matrices).
            pred_poses (list): Predicted pose list (4x4 matrices).
            result_dir (str): Directory for saving results.
            plot_key (str, optional): Unique key for plot filenames.
            logger: Logger for output.
            
        Returns:
            dict: Evaluation results with all metrics.
        """
        # Convert poses to dictionary format {frame_idx: pose}
        poses_gt_dict = {i: gt_poses[i] for i in range(len(gt_poses))}
        poses_result_dict = {i: pred_poses[i] for i in range(len(pred_poses))}
        
        # Create output directories
        plot_path_dir = os.path.join(result_dir, "plot_path")
        plot_error_dir = os.path.join(result_dir, "plot_error")
        error_dir = os.path.join(result_dir, "errors")
        
        for d in [plot_path_dir, plot_error_dir, error_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        # Calculate sequence errors
        seq_err = self._calc_sequence_errors(poses_gt_dict, poses_result_dict)
        
        if len(seq_err) == 0:
            print_log("Warning: Sequence length too short for evaluation", logger=logger)
            return {}
        
        # Save sequence errors
        error_file = os.path.join(error_dir, f"{self.test_seq_id}.txt")
        self._save_sequence_errors(seq_err, error_file)
        
        # Compute segment errors
        avg_segment_errs = self._compute_segment_error(seq_err)
        
        # Compute overall errors
        ave_t_err, ave_r_err = self._compute_overall_err(seq_err)
        
        # Compute ATE
        ate = self._compute_ATE(poses_gt_dict, poses_result_dict)
        
        # Compute RPE
        rpe_trans, rpe_rot = self._compute_RPE(poses_gt_dict, poses_result_dict)
        
        # Compute local window ATE (similar to KITTI official tool)
        local_ate_mean, local_ate_std = self._compute_local_ATE(poses_gt_dict, poses_result_dict)
        
        # Print results
        print_log(f"\nTranslational Error (%): {ave_t_err * 100:.3f}", logger=logger)
        print_log(f"Rotational Error (deg/100m): {ave_r_err / np.pi * 180 * 100:.3f}", logger=logger)
        print_log(f"ATE (m): {ate:.3f}", logger=logger)
        print_log(f"RPE Trans (m): {rpe_trans:.3f}", logger=logger)
        print_log(f"RPE Rot (deg): {rpe_rot * 180 / np.pi:.3f}", logger=logger)
        print_log(f"\nAbs Trajectory Error: {local_ate_mean:.4f}, std: {local_ate_std:.4f}", logger=logger)
        
        # Save results to file
        result_txt = os.path.join(result_dir, "result.txt")
        with open(result_txt, 'w') as f:
            f.write(f"Sequence: {self.test_seq_id}\n")
            f.write(f"Trans. err. (%): {ave_t_err * 100:.3f}\n")
            f.write(f"Rot. err. (deg/100m): {ave_r_err / np.pi * 180 * 100:.3f}\n")
            f.write(f"ATE (m): {ate:.3f}\n")
            f.write(f"RPE (m): {rpe_trans:.3f}\n")
            f.write(f"RPE (deg): {rpe_rot * 180 / np.pi:.3f}\n")
            f.write(f"Abs Trajectory Error: {local_ate_mean:.4f}, std: {local_ate_std:.4f}\n")
        
        # Generate plots
        try:
            seq_name = plot_key or str(self.test_seq_id)
            self._plot_trajectory_2d(poses_gt_dict, poses_result_dict, seq_name, plot_path_dir)
            self._plot_error(avg_segment_errs, seq_name, plot_error_dir)
            print_log(f"Plots saved to {result_dir}", logger=logger)
        except Exception as e:
            print_log(f"Warning: Plotting failed: {e}", logger=logger)
        
        # Return results
        results = {
            't_err_pct': ave_t_err * 100,
            'r_err_deg_per_100m': ave_r_err / np.pi * 180 * 100,
            'ate': ate,
            'rpe_trans': rpe_trans,
            'local_ate_mean': local_ate_mean,
            'local_ate_std': local_ate_std,
            'rpe_rot': rpe_rot * 180 / np.pi,
            'segment_errors': avg_segment_errs
        }
        
        return results
    
    def _print_summary(self, results, mode_name, logger=None):
        """Print evaluation summary.
        
        Args:
            results (dict): Evaluation results.
            mode_name (str): Evaluation mode name (e.g., "Original", "SIM3-Aligned").
            logger: Logger for output.
        """
        print_log(f"\n--- {mode_name} Pose Evaluation Summary ---", logger=logger)
        
        if results:
            print_log(f"Translational Error: {results.get('t_err_pct', 0.0):.4f}%", logger=logger)
            print_log(f"Rotational Error: {results.get('r_err_deg_per_100m', 0.0):.4f} deg/100m", logger=logger)
            print_log(f"ATE: {results.get('ate', 0.0):.4f} m", logger=logger)
            print_log(f"RPE Trans: {results.get('rpe_trans', 0.0):.4f} m", logger=logger)
            print_log(f"RPE Rot: {results.get('rpe_rot', 0.0):.4f} deg", logger=logger)
            print_log(f"Abs Trajectory Error: {results.get('local_ate_mean', 0.0):.4f}, "
                     f"std: {results.get('local_ate_std', 0.0):.4f}", logger=logger)
        else:
            print_log("No metrics available", logger=logger)
    
    # ========================================
    # Core Evaluation Methods
    # ========================================
    
    def _trajectory_distances(self, poses):
        """Compute cumulative distance for each pose w.r.t frame-0.
        
        Args:
            poses (dict): {frame_idx: 4x4 array}
            
        Returns:
            list: Cumulative distance of each pose w.r.t frame-0
        """
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        return dist
    
    def _rotation_error(self, pose_error):
        """Compute rotation error from pose error matrix.
        
        Args:
            pose_error (4x4 array): Relative pose error
            
        Returns:
            float: Rotation error in radians
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        
        # Handle numerical errors
        if np.min(d) < -1.0 or np.max(d) > 1.0:
            # Normalize rotation matrix if trace(R) is out of valid range
            u, s, vt = np.linalg.svd(pose_error[:3, :3])
            norm_r = np.dot(u, vt)
            pose_error[:3, :3] = norm_r
            a = pose_error[0, 0]
            b = pose_error[1, 1]
            c = pose_error[2, 2]
            d = 0.5 * (a + b + c - 1.0)
        
        rot_error = np.arccos(np.clip(d, -1.0, 1.0))
        return rot_error
    
    def _translation_error(self, pose_error):
        """Compute translation error from pose error matrix.
        
        Args:
            pose_error (4x4 array): Relative pose error
            
        Returns:
            float: Translation error in meters
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame index at required distance from first_frame.
        
        Args:
            dist (list): Cumulative distances
            first_frame (int): Start frame index
            length (float): Required distance
            
        Returns:
            int: End frame index, or -1 if not found
        """
        for i in range(first_frame, len(dist)):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1
    
    def _calc_sequence_errors(self, poses_gt, poses_result):
        """Calculate sequence errors over different trajectory lengths.
        
        Args:
            poses_gt (dict): {idx: 4x4 array}, ground truth poses
            poses_result (dict): {idx: 4x4 array}, predicted poses
            
        Returns:
            list: List of errors [first_frame, rot_err/len, trans_err/len, length, speed]
        """
        err = []
        dist = self._trajectory_distances(poses_gt)
        
        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self._last_frame_from_segment_length(dist, first_frame, len_)
                
                # Continue if sequence not long enough
                if last_frame == -1 or \
                        not(last_frame in poses_result.keys()) or \
                        not(first_frame in poses_result.keys()):
                    continue
                
                # Compute rotational and translational errors (relative pose error)
                pose_delta_gt = np.dot(
                    np.linalg.inv(poses_gt[first_frame]),
                    poses_gt[last_frame]
                )
                pose_delta_result = np.dot(
                    np.linalg.inv(poses_result[first_frame]),
                    poses_result[last_frame]
                )
                pose_error = np.dot(
                    np.linalg.inv(pose_delta_result),
                    pose_delta_gt
                )
                
                r_err = self._rotation_error(pose_error)
                t_err = self._translation_error(pose_error)
                
                # Compute speed (assuming 10Hz)
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)
                
                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        
        return err
    
    def _save_sequence_errors(self, err, file_name):
        """Save sequence errors to file.
        
        Args:
            err (list): Error information
            file_name (str): Output file path
        """
        with open(file_name, 'w') as fp:
            for i in err:
                line_to_write = " ".join([str(j) for j in i])
                fp.write(line_to_write + "\n")
    
    def _compute_overall_err(self, seq_err):
        """Compute average translation & rotation errors.
        
        Args:
            seq_err (list): [[first_frame, r_err, t_err, len, speed], ...]
            
        Returns:
            tuple: (ave_t_err, ave_r_err)
        """
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)
        
        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        else:
            return 0, 0
    
    def _compute_segment_error(self, seq_errs):
        """Calculate average errors for different trajectory lengths.
        
        Args:
            seq_errs (list): List of errors from _calc_sequence_errors
            
        Returns:
            dict: {length: [avg_t_err, avg_r_err], ...}
        """
        segment_errs = {}
        avg_segment_errs = {}
        
        for len_ in self.lengths:
            segment_errs[len_] = []
        
        # Collect errors by segment length
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        
        # Compute averages
        for len_ in self.lengths:
            if segment_errs[len_]:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        
        return avg_segment_errs
    
    def _compute_ATE(self, gt, pred):
        """Compute RMSE of Absolute Trajectory Error.
        
        Args:
            gt (dict): Ground-truth poses {idx: 4x4 array}
            pred (dict): Predicted poses {idx: 4x4 array}
            
        Returns:
            float: ATE (RMSE)
        """
        errors = []
        
        for i in pred:
            if i not in gt:
                continue
            
            gt_xyz = gt[i][:3, 3]
            pred_xyz = pred[i][:3, 3]
            align_err = gt_xyz - pred_xyz
            errors.append(np.sqrt(np.sum(align_err ** 2)))
        
        if len(errors) > 0:
            ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
        else:
            ate = 0.0
        
        return ate
    
    def _compute_RPE(self, gt, pred):
        """Compute Relative Pose Error.
        
        Args:
            gt (dict): Ground-truth poses {idx: 4x4 array}
            pred (dict): Predicted poses {idx: 4x4 array}
            
        Returns:
            tuple: (rpe_trans, rpe_rot) - mean RPE for translation and rotation
        """
        trans_errors = []
        rot_errors = []
        
        frame_indices = sorted(list(pred.keys()))
        
        for i in range(len(frame_indices) - 1):
            idx1 = frame_indices[i]
            idx2 = frame_indices[i + 1]
            
            if idx1 not in gt or idx2 not in gt:
                continue
            
            gt1 = gt[idx1]
            gt2 = gt[idx2]
            gt_rel = np.linalg.inv(gt1) @ gt2
            
            pred1 = pred[idx1]
            pred2 = pred[idx2]
            pred_rel = np.linalg.inv(pred1) @ pred2
            
            rel_err = np.linalg.inv(gt_rel) @ pred_rel
            
            trans_errors.append(self._translation_error(rel_err))
            rot_errors.append(self._rotation_error(rel_err))
        
        if len(trans_errors) > 0:
            rpe_trans = np.mean(np.asarray(trans_errors))
            rpe_rot = np.mean(np.asarray(rot_errors))
        else:
            rpe_trans = 0.0
            rpe_rot = 0.0
        
        return rpe_trans, rpe_rot
    
    def _compute_local_ATE(self, gt, pred, track_length=5):
        """Compute local window-based Absolute Trajectory Error.
        
        This method computes ATE using local trajectory windows with scale optimization,
        similar to the KITTI official evaluation tool. It evaluates short trajectory
        snippets and computes the average error across all windows.
        
        Args:
            gt (dict): Ground-truth poses {idx: 4x4 array}
            pred (dict): Predicted poses {idx: 4x4 array}
            track_length (int): Length of trajectory window for evaluation (default: 5)
            
        Returns:
            tuple: (mean_ate, std_ate) - mean and std of local ATE across all windows
        """
        # Convert poses to relative transformations
        frame_indices = sorted(list(pred.keys()))
        
        # Get relative poses
        gt_local_poses = []
        pred_local_poses = []
        
        for i in range(len(frame_indices) - 1):
            idx1 = frame_indices[i]
            idx2 = frame_indices[i + 1]
            
            if idx1 not in gt or idx2 not in gt:
                continue
            
            gt_rel = np.linalg.inv(gt[idx1]) @ gt[idx2]
            pred_rel = np.linalg.inv(pred[idx1]) @ pred[idx2]
            
            gt_local_poses.append(gt_rel)
            pred_local_poses.append(pred_rel)
        
        if len(gt_local_poses) < track_length:
            return 0.0, 0.0
        
        # Compute ATE for sliding windows
        ates = []
        
        for i in range(len(gt_local_poses) - track_length + 1):
            # Get window of relative poses
            gt_window = gt_local_poses[i:i + track_length]
            pred_window = pred_local_poses[i:i + track_length]
            
            # Convert to absolute trajectory (starting from origin)
            gt_xyz = self._relative_to_xyz(gt_window)
            pred_xyz = self._relative_to_xyz(pred_window)
            
            # Compute ATE with scale optimization for this window
            ate_window = self._compute_ate_with_scale(gt_xyz, pred_xyz)
            ates.append(ate_window)
        
        if len(ates) > 0:
            mean_ate = np.mean(ates)
            std_ate = np.std(ates)
        else:
            mean_ate = 0.0
            std_ate = 0.0
        
        return mean_ate, std_ate
    
    def _relative_to_xyz(self, relative_poses):
        """Convert relative poses to XYZ trajectory.
        
        Args:
            relative_poses (list): List of relative 4x4 pose matrices
            
        Returns:
            np.ndarray: Array of XYZ positions (N, 3)
        """
        xyzs = [np.array([0.0, 0.0, 0.0])]  # Start at origin
        cam_to_world = np.eye(4)
        
        for rel_pose in relative_poses:
            cam_to_world = cam_to_world @ rel_pose
            xyzs.append(cam_to_world[:3, 3])
        
        return np.array(xyzs)
    
    def _compute_ate_with_scale(self, gt_xyz, pred_xyz):
        """Compute ATE with scale optimization.
        
        Args:
            gt_xyz (np.ndarray): Ground truth XYZ trajectory (N, 3)
            pred_xyz (np.ndarray): Predicted XYZ trajectory (N, 3)
            
        Returns:
            float: RMSE after scale alignment
        """
        # Align first frame
        offset = gt_xyz[0] - pred_xyz[0]
        pred_xyz_aligned = pred_xyz + offset[None, :]
        
        # Optimize scale factor
        scale_num = np.sum(gt_xyz * pred_xyz_aligned)
        scale_den = np.sum(pred_xyz_aligned ** 2)
        
        if scale_den > 1e-8:
            scale = scale_num / scale_den
        else:
            scale = 1.0
        
        # Apply scale and compute error
        alignment_error = pred_xyz_aligned * scale - gt_xyz
        rmse = np.sqrt(np.sum(alignment_error ** 2)) / gt_xyz.shape[0]
        
        return rmse
    
    # ========================================
    # Visualization Methods
    # ========================================
    
    def _plot_trajectory_2d(self, poses_gt, poses_result, seq_name, plot_dir):
        """Plot trajectory in 2D (top-down view).
        
        Args:
            poses_gt (dict): Ground truth poses
            poses_result (dict): Predicted poses
            seq_name (str): Sequence name for plot title
            plot_dir (str): Directory to save plots
        """
        fontsize_ = 20
        
        # Extract XZ coordinates
        frame_idx_list_gt = sorted(poses_gt.keys())
        frame_idx_list_pred = sorted(poses_result.keys())
        
        pos_xz_gt = np.array([[poses_gt[idx][0, 3], poses_gt[idx][2, 3]] 
                              for idx in frame_idx_list_gt])
        pos_xz_pred = np.array([[poses_result[idx][0, 3], poses_result[idx][2, 3]] 
                                for idx in frame_idx_list_pred])
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')
        
        # Plot trajectories
        plt.plot(pos_xz_gt[:, 0], pos_xz_gt[:, 1], 'r-', label='Ground Truth', linewidth=2)
        plt.plot(pos_xz_pred[:, 0], pos_xz_pred[:, 1], 'b-', label='Prediction', linewidth=2)
        
        # Plot start point
        plt.plot(pos_xz_gt[0, 0], pos_xz_gt[0, 1], 'go', markersize=10, label='Start')
        
        # Set axis limits
        all_pos = np.vstack([pos_xz_gt, pos_xz_pred])
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        z_min, z_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        x_mid = (x_max + x_min) / 2
        z_mid = (z_max + z_min) / 2
        max_range = max(x_max - x_min, z_max - z_min) / 2 * 1.1
        
        ax.set_xlim([x_mid - max_range, x_mid + max_range])
        ax.set_ylim([z_mid - max_range, z_mid + max_range])
        
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        plt.legend(loc="upper left", prop={'size': fontsize_})
        plt.title(f'Trajectory - Sequence {seq_name}', fontsize=fontsize_)
        
        # Save figure
        png_path = os.path.join(plot_dir, f"sequence_{seq_name}.png")
        pdf_path = os.path.join(plot_dir, f"sequence_{seq_name}.pdf")
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_error(self, avg_segment_errs, seq_name, plot_dir):
        """Plot per-length translation and rotation errors.
        
        Args:
            avg_segment_errs (dict): {length: [avg_t_err, avg_r_err], ...}
            seq_name (str): Sequence name for plot title
            plot_dir (str): Directory to save plots
        """
        fontsize_ = 12
        
        # Translation error plot
        plot_x = []
        plot_y_t = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len_ in avg_segment_errs and len(avg_segment_errs[len_]) > 0:
                plot_y_t.append(avg_segment_errs[len_][0] * 100)
            else:
                plot_y_t.append(0)
        
        fig = plt.figure(figsize=(8, 6))
        plt.plot(plot_x, plot_y_t, "bs-", label="Translation Error", linewidth=2, markersize=8)
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.title(f'Translation Error - {seq_name}', fontsize=fontsize_)
        plt.grid(True, alpha=0.3)
        
        trans_pdf = os.path.join(plot_dir, f"trans_err_{seq_name}.pdf")
        trans_png = os.path.join(plot_dir, f"trans_err_{seq_name}.png")
        plt.savefig(trans_pdf, bbox_inches='tight')
        plt.savefig(trans_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Rotation error plot
        plot_y_r = []
        for len_ in self.lengths:
            if len_ in avg_segment_errs and len(avg_segment_errs[len_]) > 0:
                plot_y_r.append(avg_segment_errs[len_][1] / np.pi * 180 * 100)
            else:
                plot_y_r.append(0)
        
        fig = plt.figure(figsize=(8, 6))
        plt.plot(plot_x, plot_y_r, "rs-", label="Rotation Error", linewidth=2, markersize=8)
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.title(f'Rotation Error - {seq_name}', fontsize=fontsize_)
        plt.grid(True, alpha=0.3)
        
        rot_pdf = os.path.join(plot_dir, f"rot_err_{seq_name}.pdf")
        rot_png = os.path.join(plot_dir, f"rot_err_{seq_name}.png")
        plt.savefig(rot_pdf, bbox_inches='tight')
        plt.savefig(rot_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ========================================
    # Pose I/O and Alignment Methods
    # ========================================
    
    # ========================================
    # Pose I/O and Alignment Methods
    # ========================================
    
    def _save_poses(self, pose_list, result_dir, seq_id):
        """Save pose list to file in KITTI format.
        
        Args:
            pose_list (list): List of 4x4 pose matrices.
            result_dir (str): Directory to save poses.
            seq_id (str): Sequence ID.
        """
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        pose_path = os.path.join(result_dir, f"{seq_id}.txt")
        
        with open(pose_path, 'w') as f:
            for pose in pose_list:
                # Save first 3 rows (3x4) as required by KITTI format
                str_pose = [str(pp) for pp in pose.reshape(-1)[:12].tolist()]
                f.write(' '.join(str_pose) + '\n')
    
    def _load_poses(self, pose_dir, seq_id):
        """Load pose list from file in KITTI format.
        
        Args:
            pose_dir (str): Directory containing pose files.
            seq_id (str): Sequence ID.
            
        Returns:
            list: List of 4x4 pose matrices, or None if file not found.
        """
        pose_path = os.path.join(pose_dir, f"{seq_id}.txt")
        
        if not os.path.exists(pose_path):
            return None
        
        poses = []
        with open(pose_path, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                pose_3x4 = np.array(values).reshape(3, 4)
                pose_4x4 = np.vstack([pose_3x4, [0, 0, 0, 1]])
                poses.append(pose_4x4)
        
        return poses
    
    def _align_poses_sim3(self, pred_poses, gt_poses):
        """Align predicted poses to ground truth using SIM3 transformation.
        
        Applies Umeyama alignment (scale + rotation + translation).
        
        Args:
            pred_poses (list): List of predicted 4x4 pose matrices.
            gt_poses (list): List of ground truth 4x4 pose matrices.
            
        Returns:
            tuple: (aligned_poses, scale)
                - aligned_poses (list): List of aligned pose matrices.
                - scale (float): Scale factor.
        """
        # Extract translations
        pred_xyz = np.array([pose[:3, 3] for pose in pred_poses])
        gt_xyz = np.array([pose[:3, 3] for pose in gt_poses])
        
        # Compute SIM3 alignment
        aligned_pred_xyz, scale, R, t = self._compute_sim3_alignment(pred_xyz, gt_xyz)
        
        # Apply alignment to full poses
        aligned_poses = []
        for p_idx in range(len(pred_poses)):
            aligned_pose = pred_poses[p_idx].copy()
            # Apply rotation to rotation matrix
            aligned_pose[:3, :3] = R @ aligned_pose[:3, :3]
            # Use aligned translation
            aligned_pose[:3, 3] = aligned_pred_xyz[p_idx]
            aligned_poses.append(aligned_pose)
        
        return aligned_poses, scale
    
    def _compute_sim3_alignment(self, pred_xyz, gt_xyz):
        """Compute SIM3 (similarity) transformation alignment.
        
        Performs Umeyama alignment: scale + rotation + translation.
        
        Args:
            pred_xyz (np.ndarray): Predicted trajectory of shape (N, 3).
            gt_xyz (np.ndarray): Ground truth trajectory of shape (N, 3).
            
        Returns:
            tuple: (aligned_pred_xyz, scale, R, t)
                - aligned_pred_xyz (np.ndarray): Aligned trajectory (N, 3).
                - scale (float): Scale factor.
                - R (np.ndarray): Rotation matrix (3, 3).
                - t (np.ndarray): Translation vector (3,).
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
        
        # Compute scale (RMS ratio)
        pred_scale = np.sqrt(np.mean(np.sum(pred_centered**2, axis=1)))
        gt_scale = np.sqrt(np.mean(np.sum(gt_centered**2, axis=1)))
        scale = gt_scale / pred_scale if pred_scale > 1e-8 else 1.0
        
        # Scale the prediction
        pred_scaled = pred_centered * scale
        
        # Compute rotation using SVD (Kabsch algorithm)
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
        t = gt_centroid - np.mean((R @ pred_scaled.T).T, axis=0)
        
        # Final aligned trajectory
        aligned_pred_xyz = pred_rotated + gt_centroid
        
        return aligned_pred_xyz, scale, R, t
