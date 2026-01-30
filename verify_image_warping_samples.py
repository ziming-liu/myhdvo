#!/usr/bin/env python3
"""
验证image warping的正确性 - 使用gt_poses来warp图像
选择几个样本进行可视化验证
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_gt_pose(pose_str):
    """
    解析gt_pose字符串为4x4变换矩阵
    pose_str格式: "r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz"
    """
    values = [float(x) for x in pose_str.split()]
    assert len(values) == 12, f"Expected 12 values, got {len(values)}"
    
    # 构建4x4变换矩阵
    T = np.eye(4, dtype=np.float32)
    T[0, :] = [values[0], values[1], values[2], values[3]]
    T[1, :] = [values[4], values[5], values[6], values[7]]
    T[2, :] = [values[8], values[9], values[10], values[11]]
    
    return T


def parse_intrinsics(K_str):
    """
    解析相机内参字符串为3x3矩阵
    K_str格式: "fx 0 cx 0 0 fy cy 0 0 0 1 0"
    """
    values = [float(x) for x in K_str.split()]
    K = np.array([
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]]
    ], dtype=np.float32)
    
    return K


def read_depth(depth_path):
    """读取depth图像，vkitti2的depth是PNG格式"""
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Cannot read depth from {depth_path}")
    
    # vkitti2 depth单位是厘米，转换为米
    depth = depth.astype(np.float32) / 100.0
    
    return depth


def warp_image_with_depth(img_src, depth_src, K_src, K_tgt, T_src_to_tgt):
    """
    使用depth和变换矩阵warp图像
    参考old_vkitt2_dataset.py中的实现方式
    
    Args:
        img_src: 源图像 (H, W, 3)
        depth_src: 源图像的depth (H, W)
        K_src: 源相机内参 (3, 3)
        K_tgt: 目标相机内参 (3, 3)
        T_src_to_tgt: 从源到目标的变换矩阵 (4, 4) - world coordinate系统下的相对变换
    
    Returns:
        warped_img: warped后的图像 (H, W, 3)
        valid_mask: 有效像素的mask (H, W)
    """
    H, W = depth_src.shape
    
    # 扩展K到4x4矩阵
    K_src_44 = np.eye(4, dtype=np.float32)
    K_src_44[:3, :3] = K_src
    K_tgt_44 = np.eye(4, dtype=np.float32)
    K_tgt_44[:3, :3] = K_tgt
    
    K_inv = np.linalg.inv(K_src_44)
    
    # 创建像素网格 (参考旧代码的方式)
    y_base, x_base = np.meshgrid(
        np.linspace(0.0, H - 1.0, H),
        np.linspace(0.0, W - 1.0, W),
        indexing='ij'
    )
    
    # 构建齐次坐标 (H, W, 3)
    ones = np.ones_like(x_base)
    pixels = np.stack([x_base, y_base, ones], axis=-1)
    pixels_flat = pixels.reshape(-1, 3).T  # (3, H*W)
    
    # 添加第4维度变成齐次坐标
    pixels_homo = np.vstack([pixels_flat, np.ones((1, pixels_flat.shape[1]))])  # (4, H*W)
    
    # 反投影到3D相机坐标系
    depth_flat = depth_src.reshape(-1, 1)  # (H*W, 1)
    cam_points = K_inv @ pixels_homo  # (4, H*W)
    cam_points[:3, :] = cam_points[:3, :] * depth_flat.T  # 乘以深度
    
    # 变换到目标相机坐标系
    cam_points_tgt = T_src_to_tgt @ cam_points  # (4, H*W)
    
    # 投影到目标图像 (参考旧代码: P = K @ T)
    P = K_tgt_44 @ np.eye(4)  # 只需要K，因为已经在目标坐标系了
    pix_coords_homo = P @ cam_points_tgt  # (4, H*W)
    
    # 透视除法
    pix_coords = pix_coords_homo[:2, :] / (pix_coords_homo[2, :] + 1e-7)  # (2, H*W)
    pix_coords = pix_coords.T.reshape(H, W, 2)  # (H, W, 2)
    
    # 检查有效性
    valid_z = cam_points_tgt[2, :].reshape(H, W) > 0
    valid_u = (pix_coords[:, :, 0] >= 0) & (pix_coords[:, :, 0] < W - 1)
    valid_v = (pix_coords[:, :, 1] >= 0) & (pix_coords[:, :, 1] < H - 1)
    valid_depth = depth_src > 0
    valid_mask = valid_z & valid_u & valid_v & valid_depth
    
    # 使用cv2.remap进行warping (更高效且支持双线性插值)
    map_x = pix_coords[:, :, 0].astype(np.float32)
    map_y = pix_coords[:, :, 1].astype(np.float32)
    
    # Forward warping: 从源图像采样到目标位置
    warped_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 使用scatter方式进行forward warping
    valid_indices = np.where(valid_mask)
    for i in range(len(valid_indices[0])):
        src_y, src_x = valid_indices[0][i], valid_indices[1][i]
        tgt_x, tgt_y = map_x[src_y, src_x], map_y[src_y, src_x]
        
        tgt_x_int = int(np.round(tgt_x))
        tgt_y_int = int(np.round(tgt_y))
        
        if 0 <= tgt_x_int < W and 0 <= tgt_y_int < H:
            warped_img[tgt_y_int, tgt_x_int] = img_src[src_y, src_x]
    
    # 创建valid mask for warped image
    valid_mask_warped = np.zeros((H, W), dtype=bool)
    for i in range(len(valid_indices[0])):
        src_y, src_x = valid_indices[0][i], valid_indices[1][i]
        tgt_x, tgt_y = map_x[src_y, src_x], map_y[src_y, src_x]
        
        tgt_x_int = int(np.round(tgt_x))
        tgt_y_int = int(np.round(tgt_y))
        
        if 0 <= tgt_x_int < W and 0 <= tgt_y_int < H:
            valid_mask_warped[tgt_y_int, tgt_x_int] = True
    
    return warped_img, valid_mask_warped


def compute_photometric_error(img1, img2, mask):
    """计算光度误差"""
    if mask.sum() == 0:
        return np.inf
    
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    error = diff[mask].mean()
    return error


def visualize_warping(sample_data, sample_idx, output_dir):
    """
    可视化一个样本的warping结果
    """
    print(f"\n{'='*60}")
    print(f"Processing Sample {sample_idx}")
    print(f"{'='*60}")
    
    # 读取图像
    img0_t0_path = sample_data['image_0_paths'][0]
    img0_t1_path = sample_data['image_0_paths'][1]
    depth0_t0_path = sample_data['depth_0_paths'][0]
    
    print(f"Image 0 at t=0: {Path(img0_t0_path).name}")
    print(f"Image 0 at t=1: {Path(img0_t1_path).name}")
    print(f"Depth 0 at t=0: {Path(depth0_t0_path).name}")
    
    # 检查文件是否存在
    if not Path(img0_t0_path).exists():
        print(f"WARNING: {img0_t0_path} does not exist")
        return False
    if not Path(img0_t1_path).exists():
        print(f"WARNING: {img0_t1_path} does not exist")
        return False
    if not Path(depth0_t0_path).exists():
        print(f"WARNING: {depth0_t0_path} does not exist")
        return False
    
    img0_t0 = cv2.imread(img0_t0_path)
    img0_t1 = cv2.imread(img0_t1_path)
    depth0_t0 = read_depth(depth0_t0_path)
    
    if img0_t0 is None or img0_t1 is None:
        print("ERROR: Failed to read images")
        return False
    
    # BGR to RGB
    img0_t0 = cv2.cvtColor(img0_t0, cv2.COLOR_BGR2RGB)
    img0_t1 = cv2.cvtColor(img0_t1, cv2.COLOR_BGR2RGB)
    
    # 解析相机参数
    K = parse_intrinsics(sample_data['K_0'])
    cTw_t0 = parse_gt_pose(sample_data['gt_poses'][0])  # camera-to-world at t=0
    cTw_t1 = parse_gt_pose(sample_data['gt_poses'][1])  # camera-to-world at t=1
    
    print(f"\nCamera Intrinsics K:")
    print(K)
    print(f"\nPose at t=0 (camera-to-world):")
    print(cTw_t0)
    print(f"\nPose at t=1 (camera-to-world):")
    print(cTw_t1)
    
    # 计算相对变换: c1Tc0 = c1Tw @ wTc0 = c1Tw @ inv(c0Tw)
    # 根据旧代码的注释: T_tar == cTw (current), T_ref == rTw (reference)
    # relatvie_cTr = T_tar @ inv(T_ref) = cTw @ inv(rTw)
    # 所以要warp t=0到t=1: c1Tc0 = cTw_t1 @ inv(cTw_t0)
    c1Tc0 = cTw_t1 @ np.linalg.inv(cTw_t0)
    
    print(f"\nRelative Transform c1Tc0 (for warping t=0 -> t=1):")
    print(c1Tc0)
    print(f"Translation (x,y,z): [{c1Tc0[0,3]:.3f}, {c1Tc0[1,3]:.3f}, {c1Tc0[2,3]:.3f}]")
    
    # Warp img0_t0 到 t=1
    print("\nWarping image from t=0 to t=1...")
    warped_img, valid_mask = warp_image_with_depth(
        img0_t0, depth0_t0, K, K, c1Tc0
    )
    
    print(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.2f}%)")
    
    # 计算误差
    error = compute_photometric_error(warped_img, img0_t1, valid_mask)
    print(f"Photometric error: {error:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(img0_t0)
    axes[0, 0].set_title(f'Source Image (t=0)\n{Path(img0_t0_path).name}', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img0_t1)
    axes[0, 1].set_title(f'Target Image (t=1)\n{Path(img0_t1_path).name}', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(warped_img)
    axes[0, 2].set_title(f'Warped Image\n(t=0→t=1)', fontsize=10)
    axes[0, 2].axis('off')
    
    # 显示depth
    depth_vis = depth0_t0.copy()
    depth_vis[depth_vis > 100] = 100  # Clip for better visualization
    axes[1, 0].imshow(depth_vis, cmap='plasma')
    axes[1, 0].set_title(f'Depth (t=0)\nMax: {depth0_t0.max():.1f}m', fontsize=10)
    axes[1, 0].axis('off')
    
    # 显示差异图
    diff = np.abs(img0_t1.astype(np.float32) - warped_img.astype(np.float32))
    diff_masked = diff.copy()
    diff_masked[~valid_mask] = 0
    axes[1, 1].imshow(diff_masked.astype(np.uint8))
    axes[1, 1].set_title(f'Absolute Difference\nError: {error:.4f}', fontsize=10)
    axes[1, 1].axis('off')
    
    # 显示valid mask
    axes[1, 2].imshow(valid_mask, cmap='gray')
    axes[1, 2].set_title(f'Valid Mask\n{valid_mask.sum()} pixels', fontsize=10)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Sample {sample_idx} - Scene: {sample_data["scene"]}, Variation: {sample_data["variation"]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / f'warping_verification_sample_{sample_idx:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify image warping using ground truth poses')
    parser.add_argument('--annotation', type=str, 
                       default='/home/izi2sgh/PROJECT/hdvo/annotations/vkitti2/weather/morning/scene01.json',
                       help='Path to annotation JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/izi2sgh/PROJECT/hdvo/warping_verification_samples',
                       help='Output directory for visualizations')
    parser.add_argument('--sample_indices', type=int, nargs='+',
                       default=[0, 50, 100, 150, 200],
                       help='Sample indices to visualize')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取标注文件
    print(f"Loading annotations from: {args.annotation}")
    with open(args.annotation, 'r') as f:
        annotations = json.load(f)
    
    print(f"Total samples in annotation file: {len(annotations)}")
    print(f"Selected sample indices: {args.sample_indices}")
    
    # 处理每个样本
    success_count = 0
    for idx in args.sample_indices:
        if idx >= len(annotations):
            print(f"\nWARNING: Sample index {idx} out of range (max: {len(annotations)-1})")
            continue
        
        sample_data = annotations[idx]
        success = visualize_warping(sample_data, idx, output_dir)
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Verification Complete!")
    print(f"Successfully processed: {success_count}/{len(args.sample_indices)} samples")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
