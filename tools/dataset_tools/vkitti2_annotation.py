"""
VKitti2 dataset annotation JSON file generation script.
This script generates annotation JSON files for VKitti2 dataset in the format 
compatible with VKitti2Dataset dataloader.

VKitti2 directory structure:
data_prefix/
├── Scene01/
│   ├── clone/
│   │   ├── frames/
│   │   │   ├── rgb/
│   │   │   │   ├── Camera_0/
│   │   │   │   │   ├── rgb_00001.jpg
│   │   │   │   │   ├── ...
│   │   │   │   └── Camera_1/
│   │   │   └── depth/
│   │   │       ├── Camera_0/
│   │   │       └── Camera_1/
│   │   ├── intrinsic.txt
│   │   ├── pose.txt
│   │   └── extrinsic.txt
│   ├── 15-deg-left/
│   └── ...
├── Scene02/
└── ...

Usage:
    python tools/dataset_tools/vkitti2_annotation.py
"""

import os
import glob
import json
import numpy as np
from tqdm import tqdm


def load_vkitti2_poses(extrinsic_file_path, camera_id=0):
    """
    Load ground truth camera poses from VKitti2 extrinsic.txt file.
    
    The extrinsic.txt contains camera-to-world transformation matrices.
    Format: frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
    
    Args:
        extrinsic_file_path (str): Path to extrinsic.txt file.
        camera_id (int): Camera ID to extract poses for (0 or 1).
        
    Returns:
        list: List of pose strings (4x4 transformation matrix flattened as 12 values).
    """
    poses = []

    with open(extrinsic_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Skip header or malformed lines
        if len(parts) == 0:
            continue
        if parts[0].lower() == 'frame':
            continue
        # Expect: frame cameraID + 16 values => total >= 18 tokens
        if len(parts) < 18:
            continue

        try:
            frame_idx = int(parts[0])
            cam_id = int(parts[1])
        except ValueError:
            continue

        # Only extract poses for the specified camera
        if cam_id != camera_id:
            continue

        # Extract 4x4 matrix values (next 16 tokens)
        matrix_values = [float(v) for v in parts[2:18]]
        extrinsic_4x4 = np.array(matrix_values, dtype=np.float64).reshape(4, 4) # cTw 
        # Convert from world-to-camera  (cTw) to camera-to-world (wTc)
        extrinsic_4x4 = np.linalg.inv(extrinsic_4x4)
 
        pose_3x4 = extrinsic_4x4[:3, :]  # Take first 3 rows for 3x4 matrix
        pose_str = ' '.join([f"{float(v):.8f}" for v in pose_3x4.flatten()])
        poses.append(pose_str)

    return poses


def load_vkitti2_intrinsics(intrinsic_file_path):
    """
    Load camera intrinsics from VKitti2 intrinsic.txt file.
    
    Format: frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
    
    Args:
        intrinsic_file_path (str): Path to intrinsic.txt file.
        
    Returns:
        dict: Dictionary with frame indices as keys and intrinsic parameters as values.
              {'camera_0': [...], 'camera_1': [...]}
    """
    intrinsics = {'camera_0': {}, 'camera_1': {}}
    
    with open(intrinsic_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        # Skip header line
        if parts[0].lower() == 'frame':
            continue

        try:
            frame_idx = int(parts[0])
            camera_id = int(parts[1])
            fx = float(parts[2])
            fy = float(parts[3])
            cx = float(parts[4])
            cy = float(parts[5])
        except ValueError:
            continue

        # Store as projection matrix format: fx 0 cx 0 0 fy cy 0 0 0 1 0
        K_str = f"{fx:.8f} 0.0 {cx:.8f} 0.0 0.0 {fy:.8f} {cy:.8f} 0.0 0.0 0.0 1.0 0.0"

        camera_key = f'camera_{camera_id}'
        intrinsics[camera_key][frame_idx] = K_str
    
    return intrinsics


def compute_baseline(extrinsic_file_path):
    """
    Compute baseline between stereo cameras from extrinsic.txt file.
    
    Args:
        extrinsic_file_path (str): Path to extrinsic.txt file.
        
    Returns:
        float: Baseline distance between cameras.
    """
    # VKitti2 baseline is typically 0.532725 meters
    # You can also parse from extrinsic.txt if needed
    return 0.532725


def make_annotations(scene, variation, data_prefix, seq_len=2):
    """
    Generate annotations for a specific VKitti2 scene and variation.
    
    Args:
        scene (str): Scene name (e.g., 'Scene01', 'Scene02')
        variation (str): Variation name (e.g., 'clone', '15-deg-left', 'fog')
        data_prefix (str): Root directory of VKitti2 dataset
        seq_len (int): Sequence length (2 or 3 frames)
        
    Returns:
        list: List of annotation dictionaries.
    """
    scene_path = os.path.join(data_prefix, scene, variation)
    
    # Paths to metadata files
    intrinsic_path = os.path.join(scene_path, 'intrinsic.txt')
    extrinsic_path = os.path.join(scene_path, 'extrinsic.txt')
    
    # Check if files exist
    if not os.path.exists(intrinsic_path):
        print(f"Warning: {intrinsic_path} does not exist. Skipping {scene}/{variation}")
        return []
    if not os.path.exists(extrinsic_path):
        print(f"Warning: {extrinsic_path} does not exist. Skipping {scene}/{variation}")
        return []
    
    # Load intrinsics and camera poses
    intrinsics = load_vkitti2_intrinsics(intrinsic_path)
    gt_poses = load_vkitti2_poses(extrinsic_path, camera_id=0)  # Use camera 0 poses
    baseline = compute_baseline(extrinsic_path) if os.path.exists(extrinsic_path) else 0.532725
    
    # Get image paths
    rgb_camera0_dir = os.path.join(scene_path, 'frames', 'rgb', 'Camera_0')
    rgb_camera1_dir = os.path.join(scene_path, 'frames', 'rgb', 'Camera_1')
    depth_camera0_dir = os.path.join(scene_path, 'frames', 'depth', 'Camera_0')
    depth_camera1_dir = os.path.join(scene_path, 'frames', 'depth', 'Camera_1')
    
    camera0_images = sorted(glob.glob(os.path.join(rgb_camera0_dir, 'rgb_*.jpg')))
    camera1_images = sorted(glob.glob(os.path.join(rgb_camera1_dir, 'rgb_*.jpg')))
    camera0_depths = sorted(glob.glob(os.path.join(depth_camera0_dir, 'depth_*.png')))
    camera1_depths = sorted(glob.glob(os.path.join(depth_camera1_dir, 'depth_*.png')))
    
    if len(camera0_images) == 0:
        print(f"Warning: No images found in {rgb_camera0_dir}")
        return []

    assert len(camera0_images) == len(camera1_images), \
        f"Camera0 and Camera1 image counts don't match: {len(camera0_images)} vs {len(camera1_images)}"
    if len(gt_poses) != len(camera0_images):
        print(f"Warning: pose count ({len(gt_poses)}) != image count ({len(camera0_images)}).")
    
    num_frames = len(camera0_images)
    outputs = []
    
    # Generate annotations based on sequence length
    if seq_len == 2:
        # 2-frame sequences
        for refid, curid in tqdm(zip(range(0, num_frames - 1), range(1, num_frames)), 
                                  desc=f"{scene}/{variation}"):
            # Get intrinsics for current frames (use zero-based frame indices)
            frame_ref_idx = refid
            frame_cur_idx = curid

            # Safe fallback to the first available intrinsic if frame-specific not found
            def _get_default(camera_key):
                if len(intrinsics[camera_key]) == 0:
                    return None
                return next(iter(intrinsics[camera_key].values()))

            K_0_ref = intrinsics['camera_0'].get(frame_ref_idx, _get_default('camera_0'))
            K_1_ref = intrinsics['camera_1'].get(frame_ref_idx, _get_default('camera_1'))

            # Extract focal length from K matrix (if available)
            focal_0 = float(K_0_ref.split()[0]) if K_0_ref is not None else None
            focal_1 = float(K_1_ref.split()[0]) if K_1_ref is not None else None
            
            dict_i = dict(
                image_0_paths=[camera0_images[refid], camera0_images[curid]],
                image_1_paths=[camera1_images[refid], camera1_images[curid]],
                depth_0_paths=[camera0_depths[refid], camera0_depths[curid]] if len(camera0_depths) > 0 else None,
                depth_1_paths=[camera1_depths[refid], camera1_depths[curid]] if len(camera1_depths) > 0 else None,
                gt_poses=[gt_poses[refid], gt_poses[curid]],
                K_0=K_0_ref,
                K_1=K_1_ref,
                focal_0=str(focal_0) if focal_0 is not None else None,
                focal_1=str(focal_1) if focal_1 is not None else None,
                baseline_01=str(baseline),
                seq_len=2,
                scene=scene,
                variation=variation,
            )
            outputs.append(dict_i)
    
    elif seq_len == 3:
        # 3-frame sequences
        for preid, refid, curid in tqdm(zip(range(0, num_frames - 2), 
                                            range(1, num_frames - 1), 
                                            range(2, num_frames)),
                                        desc=f"{scene}/{variation}"):
            # Get intrinsics for current frames (zero-based)
            frame_ref_idx = refid

            def _get_default(camera_key):
                if len(intrinsics[camera_key]) == 0:
                    return None
                return next(iter(intrinsics[camera_key].values()))

            K_0_ref = intrinsics['camera_0'].get(frame_ref_idx, _get_default('camera_0'))
            K_1_ref = intrinsics['camera_1'].get(frame_ref_idx, _get_default('camera_1'))

            focal_0 = float(K_0_ref.split()[0]) if K_0_ref is not None else None
            focal_1 = float(K_1_ref.split()[0]) if K_1_ref is not None else None
            
            dict_i = dict(
                image_0_paths=[camera0_images[preid], camera0_images[refid], camera0_images[curid]],
                image_1_paths=[camera1_images[preid], camera1_images[refid], camera1_images[curid]],
                depth_0_paths=[camera0_depths[preid], camera0_depths[refid], camera0_depths[curid]] if len(camera0_depths) > 0 else None,
                depth_1_paths=[camera1_depths[preid], camera1_depths[refid], camera1_depths[curid]] if len(camera1_depths) > 0 else None,
                gt_poses=[gt_poses[preid], gt_poses[refid], gt_poses[curid]],
                K_0=K_0_ref,
                K_1=K_1_ref,
                focal_0=str(focal_0) if focal_0 is not None else None,
                focal_1=str(focal_1) if focal_1 is not None else None,
                baseline_01=str(baseline),
                seq_len=3,
                scene=scene,
                variation=variation,
            )
            outputs.append(dict_i)
    
    return outputs


def make_annotations_multi_scenes(scenes, variations, data_prefix, seq_len=2):
    """
    Generate annotations for multiple scenes and variations.
    
    Args:
        scenes (list): List of scene names
        variations (list): List of variation names
        data_prefix (str): Root directory of VKitti2 dataset
        seq_len (int): Sequence length
        
    Returns:
        list: Combined list of annotation dictionaries.
    """
    all_outputs = []
    
    for scene in scenes:
        for variation in variations:
            outputs = make_annotations(scene, variation, data_prefix, seq_len)
            all_outputs.extend(outputs)
    
    return all_outputs


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_json(save_annotation_root, name, metas):
    """
    Save annotations to JSON file.
    
    Args:
        save_annotation_root (str): Directory to save the JSON file
        name (str): Name of the JSON file (without extension)
        metas (list): List of annotation dictionaries
    """
    if not os.path.exists(save_annotation_root):
        os.makedirs(save_annotation_root)
    
    filepath = os.path.join(save_annotation_root, f"{name}.json")
    print(f'Saving {len(metas)} annotations to {filepath}')
    
    with open(filepath, mode='w') as fp:
        json.dump(metas, fp, cls=NumpyEncoder, indent=2)
    
    print(f'Successfully saved to {filepath}')


def save(save_path, name, scenes, variations, data_prefix, seq_len=2):
    """
    Main function to generate and save annotations.
    
    Args:
        save_path (str): Directory to save annotation files
        name (str): Name of the annotation file
        scenes (list): List of scene names
        variations (list): List of variation names
        data_prefix (str): Root directory of VKitti2 dataset
        seq_len (int): Sequence length (2 or 3)
    """
    metas = make_annotations_multi_scenes(scenes, variations, data_prefix, seq_len)
    make_json(save_path, name, metas)


if __name__ == '__main__':
    # Set your VKitti2 dataset root directory
    vkitti2_root = '/home/izi2sgh/PROJECT/hdvo/data_sets/vkitti2'
    
    # Define scenes and variations
    all_scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    weather_variations = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
    
    seq_len = 3 # Generate 3-frame sequences (T==3) similar to KITTI odometry
    
    # 1. Generate weather folder annotations
    print("=" * 60)
    print("Generating weather folder annotations (seq_len=3)...")
    print("=" * 60)
    
    for variation in weather_variations:
        print(f"\nProcessing variation: {variation}")
        
        # Generate annotation for each scene separately
        for scene in all_scenes:
            save(
                save_path=f'./annotations/vkitti2/weather3/{variation}',
                name=f'{scene.lower()}',
                scenes=[scene],
                variations=[variation],
                data_prefix=vkitti2_root,
                seq_len=seq_len
            )
        
        # Generate all_scenes.json for this variation
        save(
            save_path=f'./annotations/vkitti2/weather3/{variation}',
            name='all_scenes',
            scenes=all_scenes,
            variations=[variation],
            data_prefix=vkitti2_root,
            seq_len=seq_len
        )
    
    # 2. Generate dynamic folder annotations (clone variation only)
    print("\n" + "=" * 60)
    print("Generating dynamic folder annotations (seq_len=3)...")
    print("=" * 60)
    
    # Generate annotation for each scene separately
    for scene in all_scenes:
        save(
            save_path='./annotations/vkitti2/dynamic3',
            name=f'{scene.lower()}',
            scenes=[scene],
            variations=['clone'],
            data_prefix=vkitti2_root,
            seq_len=seq_len
        )
    
    # Generate all_scenes.json for dynamic
    save(
        save_path='./annotations/vkitti2/dynamic3',
        name='all_scenes',
        scenes=all_scenes,
        variations=['clone'],
        data_prefix=vkitti2_root,
        seq_len=seq_len
    )
    
    print("\n" + "=" * 60)
    print("Done! All annotations generated with seq_len=3")
    print("=" * 60)
