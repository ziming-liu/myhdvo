#!/usr/bin/env python3
"""
批量验证VKITTI2标注文件中时序图像warping的正确性
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Import functions from verify_warping.py
# Note: parse_pose_string correctly handles VKITTI2's World-to-Camera to Camera-to-World conversion
from verify_warping import parse_pose_string, parse_K_string, warp_image_with_depth, compute_photometric_error


def batch_verify_warping(annotation_file, num_samples=10, save_dir=None, stride=None):
    """
    批量验证标注文件中的图像warping
    
    Args:
        annotation_file: 标注JSON文件路径
        num_samples: 要验证的样本数量
        save_dir: 保存可视化结果的目录
        stride: 采样步长，如果为None则均匀采样
    """
    # 读取标注文件
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Total samples in dataset: {total_samples}")
    
    # 确定要测试的样本索引
    if stride is not None:
        sample_indices = list(range(0, min(num_samples * stride, total_samples), stride))
    else:
        # 均匀采样
        sample_indices = np.linspace(0, total_samples - 1, num_samples, dtype=int).tolist()
    
    print(f"Testing {len(sample_indices)} samples...")
    
    # 统计结果
    results = {
        'sample_idx': [],
        'coverage_01': [],
        'coverage_12': [],
        'coverage_02': [],
        'error_01': [],
        'error_12': [],
        'error_02': [],
        'status': []
    }
    
    # 逐个验证样本
    for idx in tqdm(sample_indices, desc="Verifying samples"):
        sample = data[idx]
        
        try:
            # 解析相机内参
            K = parse_K_string(sample['K_0'])
            
            # 读取三帧图像和深度
            images = []
            depths = []
            poses = []
            
            for i in range(3):
                # 读取图像
                img_path = sample['image_0_paths'][i]
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                
                # 读取深度图
                depth_path = sample['depth_0_paths'][i]
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise ValueError(f"Failed to load depth {depth_path}")
                depth = depth.astype(np.float32) / 100.0
                depths.append(depth)
                
                # 解析位姿
                pose = parse_pose_string(sample['gt_poses'][i])
                poses.append(pose)
            
            # 验证warping: Frame 0 -> Frame 1
            warped_01, mask_01 = warp_image_with_depth(
                images[0], depths[0], K, K, poses[0], poses[1]
            )
            coverage_01 = mask_01.sum() / (mask_01.shape[0] * mask_01.shape[1]) * 100
            error_01 = compute_photometric_error(warped_01, images[1], mask_01)
            
            # 验证warping: Frame 1 -> Frame 2
            warped_12, mask_12 = warp_image_with_depth(
                images[1], depths[1], K, K, poses[1], poses[2]
            )
            coverage_12 = mask_12.sum() / (mask_12.shape[0] * mask_12.shape[1]) * 100
            error_12 = compute_photometric_error(warped_12, images[2], mask_12)
            
            # 验证warping: Frame 0 -> Frame 2
            warped_02, mask_02 = warp_image_with_depth(
                images[0], depths[0], K, K, poses[0], poses[2]
            )
            coverage_02 = mask_02.sum() / (mask_02.shape[0] * mask_02.shape[1]) * 100
            error_02 = compute_photometric_error(warped_02, images[2], mask_02)
            
            # 判断状态
            if coverage_01 > 50 and coverage_12 > 50 and error_01 < 50 and error_12 < 50:
                status = 'GOOD'
            elif coverage_01 > 40 and coverage_12 > 40:
                status = 'ACCEPTABLE'
            else:
                status = 'POOR'
            
            # 记录结果
            results['sample_idx'].append(idx)
            results['coverage_01'].append(coverage_01)
            results['coverage_12'].append(coverage_12)
            results['coverage_02'].append(coverage_02)
            results['error_01'].append(error_01)
            results['error_12'].append(error_12)
            results['error_02'].append(error_02)
            results['status'].append(status)
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("BATCH VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    results_array = {k: np.array(v) for k, v in results.items() if k != 'status'}
    
    print(f"\nTested {len(results['sample_idx'])} samples")
    print(f"\nStatus Distribution:")
    status_counts = {}
    for s in results['status']:
        status_counts[s] = status_counts.get(s, 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({count/len(results['status'])*100:.1f}%)")
    
    print(f"\nCoverage Statistics:")
    print(f"  0→1: {np.mean(results_array['coverage_01']):.2f}% ± {np.std(results_array['coverage_01']):.2f}%")
    print(f"  1→2: {np.mean(results_array['coverage_12']):.2f}% ± {np.std(results_array['coverage_12']):.2f}%")
    print(f"  0→2: {np.mean(results_array['coverage_02']):.2f}% ± {np.std(results_array['coverage_02']):.2f}%")
    
    print(f"\nPhotometric Error Statistics:")
    print(f"  0→1: {np.mean(results_array['error_01']):.2f} ± {np.std(results_array['error_01']):.2f}")
    print(f"  1→2: {np.mean(results_array['error_12']):.2f} ± {np.std(results_array['error_12']):.2f}")
    print(f"  0→2: {np.mean(results_array['error_02']):.2f} ± {np.std(results_array['error_02']):.2f}")
    
    # 创建统计图表
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 图表1: Coverage和Error随样本变化
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Coverage
        axes[0].plot(results['sample_idx'], results_array['coverage_01'], 'o-', label='0→1', alpha=0.7)
        axes[0].plot(results['sample_idx'], results_array['coverage_12'], 's-', label='1→2', alpha=0.7)
        axes[0].plot(results['sample_idx'], results_array['coverage_02'], '^-', label='0→2', alpha=0.7)
        axes[0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Coverage (%)')
        axes[0].set_title('Warping Coverage Across Samples')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Photometric Error
        axes[1].plot(results['sample_idx'], results_array['error_01'], 'o-', label='0→1', alpha=0.7)
        axes[1].plot(results['sample_idx'], results_array['error_12'], 's-', label='1→2', alpha=0.7)
        axes[1].plot(results['sample_idx'], results_array['error_02'], '^-', label='0→2', alpha=0.7)
        axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Photometric Error')
        axes[1].set_title('Warping Photometric Error Across Samples')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'batch_verification_plots.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plots to: {save_dir / 'batch_verification_plots.png'}")
        plt.close()
        
        # 图表2: 统计分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Coverage histograms
        axes[0, 0].hist(results_array['coverage_01'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Coverage (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Coverage 0→1')
        axes[0, 0].axvline(x=np.mean(results_array['coverage_01']), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        axes[0, 1].hist(results_array['coverage_12'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Coverage (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Coverage 1→2')
        axes[0, 1].axvline(x=np.mean(results_array['coverage_12']), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        axes[0, 2].hist(results_array['coverage_02'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Coverage (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Coverage 0→2')
        axes[0, 2].axvline(x=np.mean(results_array['coverage_02']), color='r', linestyle='--', label='Mean')
        axes[0, 2].legend()
        
        # Error histograms
        axes[1, 0].hist(results_array['error_01'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Photometric Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error 0→1')
        axes[1, 0].axvline(x=np.mean(results_array['error_01']), color='r', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        axes[1, 1].hist(results_array['error_12'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Photometric Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error 1→2')
        axes[1, 1].axvline(x=np.mean(results_array['error_12']), color='r', linestyle='--', label='Mean')
        axes[1, 1].legend()
        
        axes[1, 2].hist(results_array['error_02'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Photometric Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Error 0→2')
        axes[1, 2].axvline(x=np.mean(results_array['error_02']), color='r', linestyle='--', label='Mean')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'batch_verification_distributions.png', dpi=150, bbox_inches='tight')
        print(f"Saved distributions to: {save_dir / 'batch_verification_distributions.png'}")
        plt.close()
        
        # 保存详细结果到CSV
        import csv
        csv_path = save_dir / 'batch_verification_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_Idx', 'Coverage_01', 'Coverage_12', 'Coverage_02', 
                           'Error_01', 'Error_12', 'Error_02', 'Status'])
            for i in range(len(results['sample_idx'])):
                writer.writerow([
                    results['sample_idx'][i],
                    f"{results['coverage_01'][i]:.2f}",
                    f"{results['coverage_12'][i]:.2f}",
                    f"{results['coverage_02'][i]:.2f}",
                    f"{results['error_01'][i]:.2f}",
                    f"{results['error_12'][i]:.2f}",
                    f"{results['error_02'][i]:.2f}",
                    results['status'][i]
                ])
        print(f"Saved detailed results to: {csv_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch verify image warping in VKITTI2 annotations')
    parser.add_argument('--annotation', type=str, 
                        default='annotations/vkitti2/weather/clone/scene01.json',
                        help='Path to annotation JSON file')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to verify')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride for sampling (if None, uniform sampling)')
    parser.add_argument('--save_dir', type=str, default='debug_warp_vis',
                        help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    batch_verify_warping(args.annotation, args.num_samples, args.save_dir, args.stride)
