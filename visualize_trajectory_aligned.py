#!/usr/bin/env python3
"""
Visualize Visual Odometry Trajectory Comparison with Alignment
Compare predicted trajectory with ground truth trajectory using proper alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_poses_from_txt(pose_file):
    """
    Load poses from KITTI format txt file.
    Each line contains 12 values representing a 3x4 transformation matrix.
    
    Args:
        pose_file: Path to the pose file
        
    Returns:
        poses: List of 4x4 transformation matrices
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 12:
                # Reshape to 3x4 and add bottom row
                pose = np.array(values).reshape(3, 4)
                pose = np.vstack([pose, [0, 0, 0, 1]])
                poses.append(pose)
    return np.array(poses)


def extract_trajectory_xyz(poses):
    """
    Extract x, y, z coordinates from poses.
    
    Args:
        poses: Array of 4x4 transformation matrices
        
    Returns:
        xyz: Array of xyz coordinates (N, 3)
    """
    return poses[:, :3, 3]


def align_trajectory_sim3(pred_xyz, gt_xyz):
    """
    Align predicted trajectory to ground truth using Sim3 alignment (scale + rotation + translation).
    This is similar to Umeyama alignment.
    
    Args:
        pred_xyz: Predicted trajectory (N, 3)
        gt_xyz: Ground truth trajectory (N, 3)
        
    Returns:
        aligned_pred_xyz: Aligned predicted trajectory
        scale: Scale factor
        R: Rotation matrix
        t: Translation vector
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


def align_trajectory_scale_only(pred_xyz, gt_xyz):
    """
    Align predicted trajectory to ground truth using only scale alignment.
    This matches the KITTI evaluation approach.
    
    Args:
        pred_xyz: Predicted trajectory (N, 3)
        gt_xyz: Ground truth trajectory (N, 3)
        
    Returns:
        aligned_pred_xyz: Aligned predicted trajectory
        scale: Scale factor
    """
    # Ensure same length
    min_len = min(len(pred_xyz), len(gt_xyz))
    pred_xyz = pred_xyz[:min_len]
    gt_xyz = gt_xyz[:min_len]
    
    # Align first frame
    offset = gt_xyz[0] - pred_xyz[0]
    pred_xyz_offset = pred_xyz + offset
    
    # Compute optimal scale (least squares)
    scale = np.sum(gt_xyz * pred_xyz_offset) / np.sum(pred_xyz_offset ** 2)
    if np.sum(pred_xyz_offset ** 2) == 0:
        scale = 1.0
    
    aligned_pred_xyz = pred_xyz_offset * scale
    
    return aligned_pred_xyz, scale


def calculate_metrics(pred_xyz, gt_xyz):
    """
    Calculate trajectory error metrics.
    
    Args:
        pred_xyz: Predicted trajectory (N, 3)
        gt_xyz: Ground truth trajectory (N, 3)
        
    Returns:
        Dictionary of metrics
    """
    # Align lengths
    min_len = min(len(pred_xyz), len(gt_xyz))
    pred_xyz = pred_xyz[:min_len]
    gt_xyz = gt_xyz[:min_len]
    
    # Calculate errors
    errors = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    
    metrics = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'std_error': np.std(errors),
        'rmse': np.sqrt(np.mean(errors ** 2))
    }
    
    return metrics, errors


def plot_trajectory_comparison(pred_file, gt_file, output_path=None, alignment='sim3'):
    """
    Plot trajectory comparison between predicted and ground truth.
    
    Args:
        pred_file: Path to predicted poses file
        gt_file: Path to ground truth poses file
        output_path: Optional path to save the figure
        alignment: Alignment method ('sim3', 'scale', or 'none')
    """
    # Load poses
    print(f"Loading predicted poses from: {pred_file}")
    pred_poses = load_poses_from_txt(pred_file)
    print(f"Loaded {len(pred_poses)} predicted poses")
    
    print(f"Loading ground truth poses from: {gt_file}")
    gt_poses = load_poses_from_txt(gt_file)
    print(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Extract trajectories
    pred_xyz = extract_trajectory_xyz(pred_poses)
    gt_xyz = extract_trajectory_xyz(gt_poses)
    
    # Store original for comparison
    pred_xyz_orig = pred_xyz.copy()
    
    # Apply alignment
    scale = 1.0
    if alignment == 'sim3':
        print("\nApplying Sim3 alignment (scale + rotation + translation)...")
        pred_xyz_aligned, scale, R, t = align_trajectory_sim3(pred_xyz, gt_xyz)
        print(f"Scale factor: {scale:.6f}")
        print(f"Rotation matrix:\n{R}")
        print(f"Translation: {t}")
    elif alignment == 'scale':
        print("\nApplying scale-only alignment (KITTI evaluation style)...")
        pred_xyz_aligned, scale = align_trajectory_scale_only(pred_xyz, gt_xyz)
        print(f"Scale factor: {scale:.6f}")
    else:
        print("\nNo alignment applied")
        pred_xyz_aligned = pred_xyz
    
    # Calculate metrics
    metrics_orig, errors_orig = calculate_metrics(pred_xyz_orig, gt_xyz)
    metrics_aligned, errors_aligned = calculate_metrics(pred_xyz_aligned, gt_xyz)
    
    # Calculate trajectory lengths
    pred_length = np.sum(np.linalg.norm(np.diff(pred_xyz_orig, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Subplot 1: Original trajectory (no alignment)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_xyz_orig[:, 0], pred_xyz_orig[:, 2], 'r--', linewidth=2, label='Predicted (Original)', alpha=0.8)
    ax1.plot(gt_xyz[0, 0], gt_xyz[0, 2], 'go', markersize=10, label='Start')
    ax1.plot(gt_xyz[-1, 0], gt_xyz[-1, 2], 'gs', markersize=10, label='End')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Z (m)', fontsize=12)
    ax1.set_title('Original Trajectory (No Alignment)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Subplot 2: Aligned trajectory
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax2.plot(pred_xyz_aligned[:, 0], pred_xyz_aligned[:, 2], 'b--', linewidth=2, label='Predicted', alpha=0.8)
    ax2.plot(gt_xyz[0, 0], gt_xyz[0, 2], 'go', markersize=10, label='Start')
    ax2.plot(gt_xyz[-1, 0], gt_xyz[-1, 2], 'gs', markersize=10, label='End')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.set_title(f'Aligned Trajectory ({alignment.upper()})', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Subplot 3: Error comparison over trajectory
    ax3 = plt.subplot(2, 3, 3)
    frame_indices = np.arange(len(errors_orig))
    # ax3.plot(frame_indices, errors_orig, 'r-', linewidth=1.5, alpha=0.7, label='Original')
    ax3.plot(frame_indices, errors_aligned, 'b-', linewidth=1.5, alpha=0.7)
    # ax3.axhline(y=metrics_orig['mean_error'], color='r', linestyle='--', alpha=0.5,
    #             label=f'Mean (Orig): {metrics_orig["mean_error"]:.2f} m')
    ax3.axhline(y=metrics_aligned['mean_error'], color='b', linestyle='--', alpha=0.5,
                label=f'Mean: {metrics_aligned["mean_error"]:.2f} m')
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Translation Error (m)', fontsize=12)
    ax3.set_title('Translation Error Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: 3D trajectory view (X-Y plane)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax4.plot(pred_xyz_aligned[:, 0], pred_xyz_aligned[:, 1], 'b--', linewidth=2, label='Predicted (Aligned)', alpha=0.8)
    ax4.set_xlabel('X (m)', fontsize=12)
    ax4.set_ylabel('Y (m)', fontsize=12)
    ax4.set_title('X-Y View (Aligned)', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # Subplot 5: Error histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(errors_aligned, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(x=metrics_aligned['mean_error'], color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {metrics_aligned["mean_error"]:.2f} m')
    ax5.axvline(x=metrics_aligned['median_error'], color='g', linestyle='--', linewidth=2,
                label=f'Median: {metrics_aligned["median_error"]:.2f} m')
    ax5.set_xlabel('Error (m)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Error Distribution (Aligned)', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Metrics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics_text = f"""
    Trajectory Evaluation Summary
    {'='*50}
    
    Poses: {len(pred_poses)}  |  Alignment: {alignment.upper()}
    
    ORIGINAL (No Alignment):
    • Mean Error:      {metrics_orig['mean_error']:.4f} m
    • Median Error:    {metrics_orig['median_error']:.4f} m
    • RMSE:            {metrics_orig['rmse']:.4f} m
    • Max Error:       {metrics_orig['max_error']:.4f} m
    
    ALIGNED:
    • Mean Error:      {metrics_aligned['mean_error']:.4f} m
    • Median Error:    {metrics_aligned['median_error']:.4f} m
    • RMSE:            {metrics_aligned['rmse']:.4f} m
    • Max Error:       {metrics_aligned['max_error']:.4f} m
    • Scale Factor:    {scale:.6f}
    
    TRAJECTORY LENGTH:
    • Predicted:       {pred_length:.2f} m
    • Ground Truth:    {gt_length:.2f} m
    • Ratio:           {pred_length/gt_length:.4f}
    
    Files:
    Pred: .../{os.path.basename(pred_file)}
    GT:   .../{os.path.basename(gt_file)}
    """
    
    ax6.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        output_path = pred_file.replace('.txt', f'_comparison_{alignment}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Print metrics
    print("\n" + "="*70)
    print("TRAJECTORY EVALUATION METRICS")
    print("="*70)
    print(f"{'Metric':<20} {'Original':<20} {'Aligned':<20}")
    print("-"*70)
    for key in metrics_orig.keys():
        print(f"{key:<20} {metrics_orig[key]:<20.6f} {metrics_aligned[key]:<20.6f}")
    print("-"*70)
    print(f"{'Scale factor':<20} {'-':<20} {scale:<20.6f}")
    print(f"{'Pred length (m)':<20} {pred_length:<20.2f} {pred_length:<20.2f}")
    print(f"{'GT length (m)':<20} {gt_length:<20.2f} {gt_length:<20.2f}")
    print(f"{'Length ratio':<20} {pred_length/gt_length:<20.4f} {pred_length/gt_length:<20.4f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize Visual Odometry Trajectory with Alignment')
    parser.add_argument('--pred', type=str, required=True, 
                        help='Path to predicted poses file')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth poses file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output figure (default: auto-generate)')
    parser.add_argument('--alignment', type=str, default='sim3', 
                        choices=['sim3', 'scale', 'none'],
                        help='Alignment method: sim3 (7-DoF), scale (scale-only), none (no alignment)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.pred):
        print(f"Error: Predicted poses file not found: {args.pred}")
        return
    
    if not os.path.exists(args.gt):
        print(f"Error: Ground truth poses file not found: {args.gt}")
        return
    
    # Plot comparison
    plot_trajectory_comparison(args.pred, args.gt, args.output, args.alignment)


if __name__ == '__main__':
    main()


# result file name  work_dirs/stereohdvo_posesup_coex_vkitti2_huberloss_dynamic3/pred_poses_SceneScene02_dynamic_Fri_Jan_30_00_47_46_202617473/SceneScene02_dynamic.txt
# gt file name  work_dirs/stereohdvo_posesup_coex_vkitti2_huberloss_dynamic3/gt_poses/SceneScene02_dynamic.txt


# python  visualize_trajectory_aligned.py   --pred work_dirs/stereohdvo_posesup_coex_vkitti2_huberloss_dynamic3/pred_poses_SceneScene02_dynamic_Fri_Jan_30_00_47_46_202617473/SceneScene02_dynamic.txt  --gt  work_dirs/stereohdvo_posesup_coex_vkitti2_huberloss_dynamic3/gt_poses/SceneScene02_dynamic.txt   --output   scene01_dynamic.png  --alignment  sim3 

# python  visualize_trajectory_aligned.py   --pred work_dirs/stereohdvo_posesup_coex_vkitti2_huberloss_dynamic3/pred_poses_SceneScene20_dynamic_Fri_Jan_30_01_06_25_202616177/SceneScene20_dynamic.txt  --gt  work_dirs/stereohdvo_posesup_coex_vkitti2_huberloss_dynamic3/gt_poses/SceneScene20_dynamic.txt  --output   scene20_dynamic.png  --alignment  sim3 