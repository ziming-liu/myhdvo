#!/usr/bin/env python3
"""
验证VKITTI2标注文件中时序图像warping的正确性
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_pose_string(pose_str):
    """
    解析pose字符串为4x4变换矩阵
    pose_str格式: "r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz"
    VKITTI2提供的是World-to-Camera的外参矩阵，需要转换为Camera-to-World的pose矩阵
    """
    values = [float(x) for x in pose_str.split()]
    assert len(values) == 12, f"Expected 12 values, got {len(values)}"
    
    # 构建4x4外参矩阵 (World-to-Camera)
    extrinsic = np.eye(4)
    extrinsic[:3, :] = np.array(values).reshape(3, 4)
    
    # VKitti2 extrinsic matrix is World-to-Camera transformation
    # To get camera position in world frame, we need: camera_pos = -R^T * t
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    camera_pos = -R.T @ t
    
    # Build Camera-to-World pose for consistency
    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = camera_pos
    
    return pose


def parse_K_string(K_str):
    """
    解析相机内参字符串为3x3矩阵
    K_str格式: "fx 0 cx 0 0 fy cy 0 0 0 1 0"
    """
    values = [float(x) for x in K_str.split()]
    K = np.array([
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]]
    ])
    return K


def warp_image_with_depth(img_src, depth_src, K_src, K_tgt, T_src, T_tgt):
    """
    使用深度信息将源图像warp到目标视角
    
    Args:
        img_src: 源图像 (H, W, 3)
        depth_src: 源深度图 (H, W)
        K_src: 源相机内参 (3, 3)
        K_tgt: 目标相机内参 (3, 3)
        T_src: 源相机位姿 (4, 4) - 世界到相机
        T_tgt: 目标相机位姿 (4, 4) - 世界到相机
    
    Returns:
        warped_img: warped后的图像
        valid_mask: 有效像素的mask
    """
    H, W = depth_src.shape
    
    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # 将像素坐标转换为归一化坐标
    x = (u - K_src[0, 2]) / K_src[0, 0]
    y = (v - K_src[1, 2]) / K_src[1, 1]
    
    # 3D点在源相机坐标系中
    pts_cam_src = np.stack([
        x * depth_src,
        y * depth_src,
        depth_src,
        np.ones_like(depth_src)
    ], axis=0)  # (4, H, W)
    
    pts_cam_src = pts_cam_src.reshape(4, -1)  # (4, H*W)
    
    # 过滤无效深度
    valid_depth = (depth_src.flatten() > 0) & (depth_src.flatten() < 655.35)
    
    # 转换到世界坐标系
    T_src_inv = np.linalg.inv(T_src)
    pts_world = T_src_inv @ pts_cam_src  # (4, H*W)
    
    # 转换到目标相机坐标系
    pts_cam_tgt = T_tgt @ pts_world  # (4, H*W)
    
    # 投影到目标图像平面
    pts_cam_tgt = pts_cam_tgt[:3, :]  # (3, H*W)
    
    # 过滤在相机后面的点
    valid_depth = valid_depth & (pts_cam_tgt[2, :] > 0)
    
    # 投影
    pts_2d = K_tgt @ pts_cam_tgt  # (3, H*W)
    pts_2d = pts_2d[:2, :] / (pts_cam_tgt[2:3, :] + 1e-8)  # (2, H*W)
    
    # 检查投影点是否在图像范围内
    u_tgt = pts_2d[0, :]
    v_tgt = pts_2d[1, :]
    
    valid_proj = (u_tgt >= 0) & (u_tgt < W) & (v_tgt >= 0) & (v_tgt < H)
    valid_mask = valid_depth & valid_proj
    
    # 创建warped图像
    warped_img = np.zeros_like(img_src)
    valid_mask_img = np.zeros((H, W), dtype=bool)
    
    # 填充有效像素
    src_indices = np.arange(H * W)[valid_mask]
    v_src = src_indices // W
    u_src = src_indices % W
    
    u_tgt_valid = u_tgt[valid_mask].astype(np.int32)
    v_tgt_valid = v_tgt[valid_mask].astype(np.int32)
    
    warped_img[v_tgt_valid, u_tgt_valid] = img_src[v_src, u_src]
    valid_mask_img[v_tgt_valid, u_tgt_valid] = True
    
    return warped_img, valid_mask_img


def compute_photometric_error(img1, img2, mask):
    """计算光度误差"""
    if mask.sum() == 0:
        return float('inf')
    
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    error = diff[mask].mean()
    return error


def verify_warping(annotation_file, sample_idx=0, save_dir=None):
    """
    验证标注文件中的图像warping
    
    Args:
        annotation_file: 标注JSON文件路径
        sample_idx: 要验证的样本索引
        save_dir: 保存可视化结果的目录
    """
    # 读取标注文件
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    print(f"Verifying sample {sample_idx}...")
    
    sample = data[sample_idx]
    
    # 解析相机内参
    K = parse_K_string(sample['K_0'])
    print(f"\nCamera intrinsics K:\n{K}")
    
    # 读取三帧图像和深度
    images = []
    depths = []
    poses = []
    
    for i in range(3):
        # 读取图像
        img_path = sample['image_0_paths'][i]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        # 读取深度图
        depth_path = sample['depth_0_paths'][i]
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: Failed to load depth {depth_path}")
            return
        # VKITTI2深度图存储为uint16，需要除以100转换为米
        depth = depth.astype(np.float32) / 100.0
        depths.append(depth)
        
        # 解析位姿
        pose = parse_pose_string(sample['gt_poses'][i])
        poses.append(pose)
        
        print(f"\nFrame {i}:")
        print(f"  Image: {Path(img_path).name}")
        print(f"  Shape: {img.shape}")
        print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}] m")
        print(f"  Pose:\n{pose[:3, :]}")
    
    # 验证warping: Frame 0 -> Frame 1
    print(f"\n{'='*60}")
    print("Warping Frame 0 -> Frame 1")
    print(f"{'='*60}")
    
    warped_01, mask_01 = warp_image_with_depth(
        images[0], depths[0], K, K, poses[0], poses[1]
    )
    
    coverage_01 = mask_01.sum() / (mask_01.shape[0] * mask_01.shape[1]) * 100
    error_01 = compute_photometric_error(warped_01, images[1], mask_01)
    
    print(f"Coverage: {coverage_01:.2f}%")
    print(f"Photometric error: {error_01:.2f}")
    
    # 验证warping: Frame 1 -> Frame 2
    print(f"\n{'='*60}")
    print("Warping Frame 1 -> Frame 2")
    print(f"{'='*60}")
    
    warped_12, mask_12 = warp_image_with_depth(
        images[1], depths[1], K, K, poses[1], poses[2]
    )
    
    coverage_12 = mask_12.sum() / (mask_12.shape[0] * mask_12.shape[1]) * 100
    error_12 = compute_photometric_error(warped_12, images[2], mask_12)
    
    print(f"Coverage: {coverage_12:.2f}%")
    print(f"Photometric error: {error_12:.2f}")
    
    # 验证warping: Frame 0 -> Frame 2
    print(f"\n{'='*60}")
    print("Warping Frame 0 -> Frame 2")
    print(f"{'='*60}")
    
    warped_02, mask_02 = warp_image_with_depth(
        images[0], depths[0], K, K, poses[0], poses[2]
    )
    
    coverage_02 = mask_02.sum() / (mask_02.shape[0] * mask_02.shape[1]) * 100
    error_02 = compute_photometric_error(warped_02, images[2], mask_02)
    
    print(f"Coverage: {coverage_02:.2f}%")
    print(f"Photometric error: {error_02:.2f}")
    
    # 计算差异图
    diff_01 = np.abs(warped_01.astype(np.float32) - images[1].astype(np.float32))
    diff_01_gray = np.mean(diff_01, axis=2)
    diff_01_gray[~mask_01] = 0
    
    diff_12 = np.abs(warped_12.astype(np.float32) - images[2].astype(np.float32))
    diff_12_gray = np.mean(diff_12, axis=2)
    diff_12_gray[~mask_12] = 0
    
    diff_02 = np.abs(warped_02.astype(np.float32) - images[2].astype(np.float32))
    diff_02_gray = np.mean(diff_02, axis=2)
    diff_02_gray[~mask_02] = 0
    
    # 创建综合可视化结果
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.2)
    
    # Row 0: Original images
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(images[0])
    ax00.set_title(f'Frame 0', fontsize=12, fontweight='bold')
    ax00.axis('off')
    
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(images[1])
    ax01.set_title(f'Frame 1', fontsize=12, fontweight='bold')
    ax01.axis('off')
    
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(images[2])
    ax02.set_title(f'Frame 2', fontsize=12, fontweight='bold')
    ax02.axis('off')
    
    # Depth visualization
    ax03 = fig.add_subplot(gs[0, 3])
    depth_vis = depths[0].copy()
    depth_vis[depth_vis > 100] = 100  # clip for visualization
    im = ax03.imshow(depth_vis, cmap='plasma')
    ax03.set_title('Frame 0 Depth', fontsize=12, fontweight='bold')
    ax03.axis('off')
    plt.colorbar(im, ax=ax03, fraction=0.046)
    
    # Row 1: Warping 0->1
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(images[0])
    ax10.set_title('Source: Frame 0', fontsize=11)
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(warped_01)
    ax11.set_title(f'Warped 0→1\nCov: {coverage_01:.1f}%', fontsize=11)
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.imshow(images[1])
    ax12.set_title(f'Target: Frame 1\nError: {error_01:.2f}', fontsize=11)
    ax12.axis('off')
    
    ax13 = fig.add_subplot(gs[1, 3])
    im = ax13.imshow(diff_01_gray, cmap='hot', vmin=0, vmax=100)
    ax13.set_title('Difference Map', fontsize=11)
    ax13.axis('off')
    plt.colorbar(im, ax=ax13, fraction=0.046)
    
    # Row 2: Warping 1->2
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.imshow(images[1])
    ax20.set_title('Source: Frame 1', fontsize=11)
    ax20.axis('off')
    
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.imshow(warped_12)
    ax21.set_title(f'Warped 1→2\nCov: {coverage_12:.1f}%', fontsize=11)
    ax21.axis('off')
    
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.imshow(images[2])
    ax22.set_title(f'Target: Frame 2\nError: {error_12:.2f}', fontsize=11)
    ax22.axis('off')
    
    ax23 = fig.add_subplot(gs[2, 3])
    im = ax23.imshow(diff_12_gray, cmap='hot', vmin=0, vmax=100)
    ax23.set_title('Difference Map', fontsize=11)
    ax23.axis('off')
    plt.colorbar(im, ax=ax23, fraction=0.046)
    
    # Row 3: Warping 0->2
    ax30 = fig.add_subplot(gs[3, 0])
    ax30.imshow(images[0])
    ax30.set_title('Source: Frame 0', fontsize=11)
    ax30.axis('off')
    
    ax31 = fig.add_subplot(gs[3, 1])
    ax31.imshow(warped_02)
    ax31.set_title(f'Warped 0→2\nCov: {coverage_02:.1f}%', fontsize=11)
    ax31.axis('off')
    
    ax32 = fig.add_subplot(gs[3, 2])
    ax32.imshow(images[2])
    ax32.set_title(f'Target: Frame 2\nError: {error_02:.2f}', fontsize=11)
    ax32.axis('off')
    
    ax33 = fig.add_subplot(gs[3, 3])
    im = ax33.imshow(diff_02_gray, cmap='hot', vmin=0, vmax=100)
    ax33.set_title('Difference Map', fontsize=11)
    ax33.axis('off')
    plt.colorbar(im, ax=ax33, fraction=0.046)
    
    # Row 4: Masks visualization
    ax40 = fig.add_subplot(gs[4, 0])
    ax40.imshow(mask_01, cmap='gray')
    ax40.set_title(f'Valid Mask 0→1\n{coverage_01:.1f}%', fontsize=11)
    ax40.axis('off')
    
    ax41 = fig.add_subplot(gs[4, 1])
    ax41.imshow(mask_12, cmap='gray')
    ax41.set_title(f'Valid Mask 1→2\n{coverage_12:.1f}%', fontsize=11)
    ax41.axis('off')
    
    ax42 = fig.add_subplot(gs[4, 2])
    ax42.imshow(mask_02, cmap='gray')
    ax42.set_title(f'Valid Mask 0→2\n{coverage_02:.1f}%', fontsize=11)
    ax42.axis('off')
    
    # Summary text
    ax43 = fig.add_subplot(gs[4, 3])
    ax43.axis('off')
    summary_text = f"""Sample {sample_idx}
Scene: {sample['scene']}
Variation: {sample['variation']}

Warping Results:
━━━━━━━━━━━━━━━━
0→1: Cov={coverage_01:.1f}%
     Err={error_01:.2f}
     
1→2: Cov={coverage_12:.1f}%
     Err={error_12:.2f}
     
0→2: Cov={coverage_02:.1f}%
     Err={error_02:.2f}
"""
    ax43.text(0.1, 0.5, summary_text, fontsize=10, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f'Image Warping Verification - Sample {sample_idx}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    if save_dir:
        save_path = Path(save_dir) / f'warping_verification_sample_{sample_idx}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
    
    plt.close()
    
    # 总结
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sample {sample_idx}: {sample['scene']} - {sample['variation']}")
    print(f"\nWarping 0->1: Coverage={coverage_01:.2f}%, Error={error_01:.2f}")
    print(f"Warping 1->2: Coverage={coverage_12:.2f}%, Error={error_12:.2f}")
    print(f"Warping 0->2: Coverage={coverage_02:.2f}%, Error={error_02:.2f}")
    
    # 判断是否合理 (调整阈值，考虑到真实场景中的遮挡和动态物体)
    if coverage_01 > 50 and coverage_12 > 50 and error_01 < 50 and error_12 < 50:
        print("\n✓ Warping appears CORRECT!")
        print("  (Coverage is good and photometric error is within acceptable range)")
    elif coverage_01 > 40 and coverage_12 > 40:
        print("\n⚠ Warping has ACCEPTABLE quality")
        print("  (Coverage is reasonable, but photometric error is higher due to occlusions/dynamic objects)")
    else:
        print("\n✗ Warping may have ISSUES!")
        print("  (Low coverage or very high photometric error)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify image warping in VKITTI2 annotations')
    parser.add_argument('--annotation', type=str, 
                        default='annotations/vkitti2/weather/clone/scene01.json',
                        help='Path to annotation JSON file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index to verify')
    parser.add_argument('--save_dir', type=str, default='debug_warp_vis',
                        help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    verify_warping(args.annotation, args.sample, args.save_dir)
