"""
Visualize VKitti2 camera trajectories in Bird's Eye View (BEV).
This script reads the extrinsic.txt files from VKitti2 scenes and plots the camera trajectories.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob


def load_vkitti2_trajectory(extrinsic_file, camera_id=0):
    """
    Load camera trajectory from VKitti2 extrinsic.txt file.
    
    The extrinsic matrix represents the camera-to-world transformation:
    [R | t] where R is rotation and t is translation
    
    Args:
        extrinsic_file (str): Path to extrinsic.txt file
        camera_id (int): Camera ID (0 or 1)
        
    Returns:
        np.ndarray: Trajectory positions (N, 3) - x, y, z coordinates
        np.ndarray: Trajectory poses (N, 4, 4) - full transformation matrices
    """
    positions = []
    poses = []
    
    with open(extrinsic_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        
        # Skip header
        if parts[0] == 'frame':
            continue
        
        if len(parts) < 17:
            continue
        
        frame_idx = int(parts[0])
        cam_id = int(parts[1])
        
        # Only extract for specified camera
        if cam_id != camera_id:
            continue
        
        # Parse 4x4 transformation matrix
        # Header: frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
        # Extrinsic = [r1,1 r1,2 r1,3 t1  ]
        #             [r2,1 r2,2 r2,3 t2  ]
        #             [r3,1 r3,2 r3,3 t3  ]
        #             [0    0    0    1   ]
        matrix_values = [float(v) for v in parts[2:18]]
        
        # Reshape to 4x4 matrix (row-major order)
        extrinsic = np.array(matrix_values).reshape(4, 4)
        
        # VKitti2 extrinsic matrix is World-to-Camera transformation
        # To get camera position in world frame, we need: camera_pos = -R^T * t
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        camera_pos = -R.T @ t
        
        # Build Camera-to-World pose for consistency
        pose = np.eye(4)
        pose[:3, :3] = R.T
        pose[:3, 3] = camera_pos
        poses.append(pose)
        
        # Position: camera position in world coordinates
        positions.append(camera_pos)
    
    return np.array(positions), np.array(poses)


def plot_trajectory_bev(ax, positions, label, color, linestyle='-', linewidth=2, alpha=0.8):
    """
    Plot trajectory in bird's eye view (top-down, XZ plane).
    
    In VKitti2 coordinate system:
    - X (t1): horizontal (left-right)
    - Y (t2): height (up-down) 
    - Z (t3): forward (front-back)
    
    BEV should be X-Z plane (horizontal view from above).
    
    Args:
        ax: Matplotlib axis
        positions (np.ndarray): Positions (N, 3) where columns are [X, Y, Z]
        label (str): Trajectory label
        color: Line color
        linestyle (str): Line style
        linewidth (float): Line width
        alpha (float): Transparency
    """
    # Plot XZ trajectory (bird's eye view: X horizontal, Z forward)
    # positions[:, 0] is X (left-right)
    # positions[:, 2] is Z (forward-backward)
    ax.plot(positions[:, 0], positions[:, 2], 
            color=color, linestyle=linestyle, linewidth=linewidth, 
            alpha=alpha, label=label)
    
    # Mark start point
    ax.scatter(positions[0, 0], positions[0, 2], 
               color=color, marker='o', s=100, zorder=5, 
               edgecolors='black', linewidths=1.5)
    
    # Mark end point
    ax.scatter(positions[-1, 0], positions[-1, 2], 
               color=color, marker='s', s=100, zorder=5,
               edgecolors='black', linewidths=1.5)


def visualize_all_scenes(data_root, output_path='vkitti2_trajectories_bev.png'):
    """
    Visualize trajectories for all VKitti2 scenes.
    
    Args:
        data_root (str): Root directory of VKitti2 dataset
        output_path (str): Output image path
    """
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    variations = ['clone', '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
                  'fog', 'morning', 'overcast', 'rain', 'sunset']
    
    # Color palette for different variations
    variation_colors = {
        'clone': '#1f77b4',
        '15-deg-left': '#ff7f0e',
        '15-deg-right': '#2ca02c',
        '30-deg-left': '#d62728',
        '30-deg-right': '#9467bd',
        'fog': '#8c564b',
        'morning': '#e377c2',
        'overcast': '#7f7f7f',
        'rain': '#bcbd22',
        'sunset': '#17becf'
    }
    
    # Create figure with subplots for each scene
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    
    print("=" * 80)
    print("Visualizing VKitti2 Camera Trajectories")
    print("=" * 80)
    
    for scene_idx, scene in enumerate(scenes):
        ax = axes[scene_idx]
        ax.set_title(f'{scene} Trajectories', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        print(f"\n{scene}:")
        
        trajectories_found = False
        
        for variation in variations:
            extrinsic_path = os.path.join(data_root, scene, variation, 'extrinsic.txt')
            
            if not os.path.exists(extrinsic_path):
                continue
            
            try:
                positions, poses = load_vkitti2_trajectory(extrinsic_path, camera_id=0)
                
                if len(positions) > 0:
                    trajectories_found = True
                    
                    # Plot trajectory
                    color = variation_colors.get(variation, '#000000')
                    plot_trajectory_bev(ax, positions, variation, color)
                    
                    # Print statistics
                    trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                    print(f"  {variation:15s}: {len(positions):4d} frames, "
                          f"length: {trajectory_length:7.2f}m, "
                          f"range: X[{positions[:, 0].min():7.2f}, {positions[:, 0].max():7.2f}], "
                          f"Y[{positions[:, 1].min():7.2f}, {positions[:, 1].max():7.2f}]")
                
            except Exception as e:
                print(f"  {variation:15s}: Error - {str(e)}")
        
        if trajectories_found:
            ax.legend(loc='best', fontsize=8, ncol=2)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    # Add overall title
    fig.suptitle('VKitti2 Camera Trajectories - Bird\'s Eye View', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n" + "=" * 80)
    print(f"Visualization saved to: {output_path}")
    print("=" * 80)
    
    return fig


def visualize_scene_comparison(data_root, output_path='vkitti2_scenes_comparison.png'):
    """
    Compare 'clone' variation trajectories across all scenes in one plot.
    
    Args:
        data_root (str): Root directory of VKitti2 dataset
        output_path (str): Output image path
    """
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    scene_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('VKitti2 Scenes Comparison (Clone Variation)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    print("\n" + "=" * 80)
    print("Scene Comparison (Clone Variation):")
    print("=" * 80)
    
    for scene, color in zip(scenes, scene_colors):
        extrinsic_path = os.path.join(data_root, scene, 'clone', 'extrinsic.txt')
        
        if not os.path.exists(extrinsic_path):
            print(f"{scene}: Not found")
            continue
        
        try:
            positions, poses = load_vkitti2_trajectory(extrinsic_path, camera_id=0)
            
            if len(positions) > 0:
                plot_trajectory_bev(ax, positions, scene, color, linewidth=2.5)
                
                trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                print(f"{scene}: {len(positions):4d} frames, length: {trajectory_length:7.2f}m")
        
        except Exception as e:
            print(f"{scene}: Error - {str(e)}")
    
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nComparison plot saved to: {output_path}")
    print("=" * 80)
    
    return fig


def visualize_3d_trajectories(data_root, output_path='vkitti2_trajectories_3d.png'):
    """
    Visualize trajectories in 3D view.
    
    Args:
        data_root (str): Root directory of VKitti2 dataset
        output_path (str): Output image path
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    scene_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('VKitti2 3D Camera Trajectories (Clone Variation)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    
    for scene, color in zip(scenes, scene_colors):
        extrinsic_path = os.path.join(data_root, scene, 'clone', 'extrinsic.txt')
        
        if not os.path.exists(extrinsic_path):
            continue
        
        try:
            positions, poses = load_vkitti2_trajectory(extrinsic_path, camera_id=0)
            
            if len(positions) > 0:
                # Plot 3D trajectory
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                       color=color, linewidth=2, alpha=0.8, label=scene)
                
                # Mark start
                ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                          color=color, marker='o', s=100, edgecolors='black', linewidths=1.5)
                
                # Mark end
                ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                          color=color, marker='s', s=100, edgecolors='black', linewidths=1.5)
        
        except Exception as e:
            print(f"Error loading {scene}: {str(e)}")
    
    ax.legend(loc='best', fontsize=12)
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"3D visualization saved to: {output_path}")
    
    return fig


def visualize_individual_scenes(data_root, output_dir='./'):
    """
    Generate individual BEV plots for each scene (clone variation only).
    Each scene gets its own figure with equal aspect ratio.
    
    Args:
        data_root (str): Root directory of VKitti2 dataset
        output_dir (str): Output directory for images
    """
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    scene_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    print("=" * 80)
    print("Generating Individual Scene BEV Plots (Clone Variation Only)")
    print("=" * 80)
    
    for scene, color in zip(scenes, scene_colors):
        extrinsic_path = os.path.join(data_root, scene, 'clone', 'extrinsic.txt')
        
        if not os.path.exists(extrinsic_path):
            print(f"\n{scene}: Extrinsic file not found, skipping...")
            continue
        
        try:
            # Load trajectory
            positions, poses = load_vkitti2_trajectory(extrinsic_path, camera_id=0)
            
            if len(positions) == 0:
                print(f"\n{scene}: No trajectory data, skipping...")
                continue
            
            # Create individual figure (square aspect)
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Calculate coordinate ranges first (needed for arrow sizing)
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
            x_range = x_max - x_min
            z_range = z_max - z_min
            
            # Make ranges equal (square plot with equal axis lengths)
            max_range = max(x_range, z_range)
            x_center = (x_min + x_max) / 2
            z_center = (z_min + z_max) / 2
            
            # Plot trajectory (XZ plane for BEV)
            ax.plot(positions[:, 0], positions[:, 2], 
                   color=color, linewidth=3, alpha=0.8, label=f'{scene} Trajectory')
            
            # Mark start point (circle)
            ax.scatter(positions[0, 0], positions[0, 2], 
                      color='green', marker='o', s=200, zorder=5, 
                      edgecolors='black', linewidths=2, label='Start')
            
            # Mark end point (square)
            ax.scatter(positions[-1, 0], positions[-1, 2], 
                      color='red', marker='s', s=200, zorder=5,
                      edgecolors='black', linewidths=2, label='End')
            
            # Calculate statistics
            trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            
            # Set equal ranges centered around data
            margin = max_range * 0.1  # 10% margin
            axis_half_range = max_range / 2 + margin
            
            ax.set_xlim(x_center - axis_half_range, x_center + axis_half_range)
            ax.set_ylim(z_center - axis_half_range, z_center + axis_half_range)
            
            # Set title and labels
            ax.set_title(f'{scene} Camera Trajectory - Bird\'s Eye View\n'
                        f'Frames: {len(positions)}, Length: {trajectory_length:.2f}m',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('X (meters) - Horizontal', fontsize=14, fontweight='bold')
            ax.set_ylabel('Z (meters) - Forward', fontsize=14, fontweight='bold')
            
            # Set equal aspect ratio (force same scale for x and y)
            ax.set_aspect('equal', adjustable='box')
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
            ax.legend(loc='best', fontsize=14, framealpha=0.9)
            
            # Add statistics text box
            stats_text = f'X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m\n'
            stats_text += f'Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m (height)\n'
            stats_text += f'Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, f'vkitti2_{scene}_clone_bev.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print summary
            print(f"\n{scene}:")
            print(f"  Frames: {len(positions)}")
            print(f"  Trajectory length: {trajectory_length:.2f} m")
            print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
            print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
            print(f"  Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
            print(f"  ✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"\n{scene}: Error - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("All individual scene plots completed!")
    print("=" * 80)


if __name__ == '__main__':
    # VKitti2 dataset root
    data_root = '/home/izi2sgh/PROJECT/hdvo/data_sets/vkitti2'
    
    # Check if dataset exists
    if not os.path.exists(data_root):
        print(f"Error: VKitti2 dataset not found at {data_root}")
        exit(1)
    
    # Generate individual scene plots (clone variation only)
    print("\nGenerating individual BEV plots for each scene...")
    visualize_individual_scenes(data_root, output_dir='./')
    
    print("\n✓ All visualizations completed!")
