"""
Test script for VKitti2Dataset dataloader.
This script tests whether the VKitti2Dataset can correctly load data from the generated annotations.
"""

import os
import sys
import numpy as np

# Add hdvo to path
sys.path.insert(0, os.path.dirname(__file__))

from hdvo.datasets import VKitti2Dataset


def test_vkitti2_dataset():
    """Test VKitti2Dataset dataloader with generated annotations."""
    
    print("=" * 80)
    print("Testing VKitti2Dataset Dataloader")
    print("=" * 80)
    
    # Configuration
    data_prefix = '/home/izi2sgh/PROJECT/hdvo/data_sets/vkitti2'
    ann_file = './annotations/vkitti2/vkitti2_train_scene01_clone.json'
    scene = 'Scene01'
    variation = 'clone'
    target_size = (1024, 320)  # (width, height)
    
    # Create dataset
    print(f"\n1. Creating VKitti2Dataset...")
    print(f"   - Data prefix: {data_prefix}")
    print(f"   - Annotation file: {ann_file}")
    print(f"   - Scene: {scene}, Variation: {variation}")
    print(f"   - Target size: {target_size}")
    
    dataset = VKitti2Dataset(
        data_prefix=data_prefix,
        ann_file=ann_file,
        scene=scene,
        variation=variation,
        target_size=target_size,
        pipeline=[],  # No augmentation for testing
        test_mode=False
    )
    
    print(f"\n2. Dataset loaded successfully!")
    print(f"   - Total samples: {len(dataset)}")
    
    # Test loading first sample
    print(f"\n3. Testing data loading...")
    sample_idx = 0
    data = dataset[sample_idx]
    
    print(f"\n4. Sample {sample_idx} information:")
    print(f"   Keys in data dict: {list(data.keys())}")
    
    # Check images
    if 'left_frame_paths' in data:
        print(f"\n5. Image information:")
        print(f"   - Left image paths: {data['left_frame_paths']}")
        print(f"   - Right image paths: {data['right_frame_paths']}")
        
        # Check first image
        img0_path = data['left_frame_paths'][0]
        if os.path.exists(img0_path):
            print(f"   - First left image exists: True")
            print(f"   - Path: {img0_path}")
        else:
            print(f"   - WARNING: Image file not found: {img0_path}")
    elif 'image_0_paths' in data:
        print(f"\n5. Image information:")
        print(f"   - Left image paths: {data['image_0_paths']}")
        print(f"   - Right image paths: {data['image_1_paths']}")
        
        # Check first image
        img0_path = data['image_0_paths'][0]
        if os.path.exists(img0_path):
            print(f"   - First left image exists: True")
            print(f"   - Path: {img0_path}")
        else:
            print(f"   - WARNING: Image file not found: {img0_path}")
    
    # Check depth maps
    if 'depth_0_paths' in data and data['depth_0_paths'] is not None:
        print(f"\n6. Depth information:")
        print(f"   - Left depth paths: {data['depth_0_paths']}")
        
        depth0_path = data['depth_0_paths'][0]
        if os.path.exists(depth0_path):
            print(f"   - First left depth exists: True")
            print(f"   - Path: {depth0_path}")
        else:
            print(f"   - WARNING: Depth file not found: {depth0_path}")
    
    # Check camera intrinsics
    if 'k_left' in data:
        print(f"\n7. Camera intrinsics:")
        print(f"   - k_left: {data['k_left']}")
        print(f"   - K_right: {data['K_right']}")
        print(f"   - focal: {data['focal']}")
        print(f"   - focal_right: {data['focal_right']}")
        print(f"   - baseline: {data['baseline']}")
        if 'intrinsics' in data:
            print(f"   - intrinsics shape: {np.array(data['intrinsics']).shape}")
    elif 'K_0' in data:
        print(f"\n7. Camera intrinsics:")
        print(f"   - K_0: {data['K_0']}")
        print(f"   - K_1: {data['K_1']}")
        print(f"   - focal_0: {data['focal_0']}")
        print(f"   - focal_1: {data['focal_1']}")
        print(f"   - baseline_01: {data['baseline_01']}")
    
    # Check ground truth poses
    if 'pose' in data:
        print(f"\n8. Ground truth poses:")
        if isinstance(data['pose'], list):
            for i, pose in enumerate(data['pose']):
                if isinstance(pose, str):
                    pose_values = [float(v) for v in pose.split()]
                else:
                    pose_values = pose
                print(f"   - Pose {i}: {len(pose_values)} values")
                # Reshape to 3x4 matrix for display
                pose_matrix = np.array(pose_values).reshape(3, 4)
                print(f"     {pose_matrix[0]}")
                print(f"     {pose_matrix[1]}")
                print(f"     {pose_matrix[2]}")
        elif isinstance(data['pose'], np.ndarray):
            print(f"   - Pose shape: {data['pose'].shape}")
            print(f"   - Pose:\n{data['pose']}")
    elif 'gt_poses' in data:
        print(f"\n8. Ground truth poses:")
        for i, pose in enumerate(data['gt_poses']):
            pose_values = [float(v) for v in pose.split()]
            print(f"   - Pose {i}: {len(pose_values)} values")
            # Reshape to 3x4 matrix for display
            pose_matrix = np.array(pose_values).reshape(3, 4)
            print(f"     {pose_matrix[0]}")
            print(f"     {pose_matrix[1]}")
            print(f"     {pose_matrix[2]}")
    
    # Test multiple samples
    print(f"\n9. Testing multiple samples...")
    test_indices = [0, 10, 50, 100, 200]
    for idx in test_indices:
        if idx < len(dataset):
            try:
                data = dataset[idx]
                print(f"   ✓ Sample {idx}: Successfully loaded")
            except Exception as e:
                print(f"   ✗ Sample {idx}: Failed - {str(e)}")
        else:
            print(f"   - Sample {idx}: Out of range (dataset size: {len(dataset)})")
    
    print(f"\n" + "=" * 80)
    print("VKitti2Dataset Test Completed!")
    print("=" * 80)
    
    return dataset


if __name__ == '__main__':
    # Test the dataset
    dataset = test_vkitti2_dataset()
