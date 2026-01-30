#!/usr/bin/env python3
"""
Visualize Visual Odometry Trajectory Comparison
Compare predicted trajectory with ground truth trajectory
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


def extract_trajectory_xy(poses):
    """
    Extract x, y coordinates from poses for bird's eye view.
    
    Args:
        poses: Array of 4x4 transformation matrices
        
    Returns:
        x, z: Arrays of x and z coordinates (bird's eye view)
    """
    x = poses[:, 0, 3]
    z = poses[:, 2, 3]
    return x, z


def calculate_metrics(pred_poses, gt_poses):
    """
    Calculate trajectory error metrics.
    
    Args:
        pred_poses: Predicted poses
        gt_poses: Ground truth poses
        
    Returns:
        Dictionary of metrics
    """
    # Extract positions
    pred_pos = pred_poses[:, :3, 3]
    gt_pos = gt_poses[:, :3, 3]
    
    # Align lengths
    min_len = min(len(pred_pos), len(gt_pos))
    pred_pos = pred_pos[:min_len]
    gt_pos = gt_pos[:min_len]
    
    # Calculate errors
    errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
    
    metrics = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'std_error': np.std(errors),
        'rmse': np.sqrt(np.mean(errors ** 2))
    }
    
    return metrics, errors


def plot_trajectory_comparison(pred_file, gt_file, output_path=None):
    """
    Plot trajectory comparison between predicted and ground truth.
    
    Args:
        pred_file: Path to predicted poses file
        gt_file: Path to ground truth poses file
        output_path: Optional path to save the figure
    """
    # Load poses
    print(f"Loading predicted poses from: {pred_file}")
    pred_poses = load_poses_from_txt(pred_file)
    print(f"Loaded {len(pred_poses)} predicted poses")
    
    print(f"Loading ground truth poses from: {gt_file}")
    gt_poses = load_poses_from_txt(gt_file)
    print(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Extract trajectories
    pred_x, pred_z = extract_trajectory_xy(pred_poses)
    gt_x, gt_z = extract_trajectory_xy(gt_poses)
    
    # Calculate metrics
    metrics, errors = calculate_metrics(pred_poses, gt_poses)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Bird's eye view trajectory
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(gt_x, gt_z, 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_x, pred_z, 'r--', linewidth=2, label='Predicted', alpha=0.8)
    ax1.plot(gt_x[0], gt_z[0], 'go', markersize=10, label='Start')
    ax1.plot(gt_x[-1], gt_z[-1], 'gs', markersize=10, label='End')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Z (m)', fontsize=12)
    ax1.set_title('Trajectory Comparison (Bird\'s Eye View)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Subplot 2: Error over trajectory
    ax2 = plt.subplot(1, 3, 2)
    frame_indices = np.arange(len(errors))
    ax2.plot(frame_indices, errors, 'b-', linewidth=1.5)
    ax2.axhline(y=metrics['mean_error'], color='r', linestyle='--', 
                label=f'Mean: {metrics["mean_error"]:.3f} m')
    ax2.fill_between(frame_indices, 0, errors, alpha=0.3)
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Translation Error (m)', fontsize=12)
    ax2.set_title('Translation Error over Trajectory', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Metrics summary
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    metrics_text = f"""
    Trajectory Metrics Summary
    {'='*40}
    
    Number of Poses: {len(pred_poses)}
    
    Translation Errors:
    • Mean Error:      {metrics['mean_error']:.4f} m
    • Median Error:    {metrics['median_error']:.4f} m
    • RMSE:            {metrics['rmse']:.4f} m
    • Std Dev:         {metrics['std_error']:.4f} m
    • Max Error:       {metrics['max_error']:.4f} m
    • Min Error:       {metrics['min_error']:.4f} m
    
    Total Distance (GT): {np.sum(np.linalg.norm(np.diff(gt_poses[:, :3, 3], axis=0), axis=1)):.2f} m
    
    Files:
    Pred: {os.path.basename(pred_file)}
    GT:   {os.path.basename(gt_file)}
    """
    
    ax3.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        output_path = pred_file.replace('.txt', '_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Print metrics
    print("\n" + "="*50)
    print("Trajectory Evaluation Metrics")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f} m")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Visualize Visual Odometry Trajectory Comparison')
    parser.add_argument('--pred', type=str, required=True, 
                        help='Path to predicted poses file')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth poses file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output figure (default: auto-generate)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.pred):
        print(f"Error: Predicted poses file not found: {args.pred}")
        return
    
    if not os.path.exists(args.gt):
        print(f"Error: Ground truth poses file not found: {args.gt}")
        return
    
    # Plot comparison
    plot_trajectory_comparison(args.pred, args.gt, args.output)


if __name__ == '__main__':
    main()
