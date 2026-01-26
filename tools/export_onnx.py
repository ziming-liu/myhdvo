"""
ONNX Export Script for HDVO Model
Author: ACENTAURI team
Date: 2025-01-27
"""
import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint

from hdvo.models import build_model
from hdvo.datasets import build_dataloader, build_dataset


class DepthNetWrapper(nn.Module):
    """Wrapper for depth network to make it ONNX-exportable"""
    def __init__(self, depth_net):
        super().__init__()
        self.depth_net = depth_net
    
    def forward(self, left_img, right_img):
        """
        Args:
            left_img: [B, C, H, W]
            right_img: [B, C, H, W]
        Returns:
            disp: [B, 1, H, W]
        """
        disps = self.depth_net(left_img, right_img)
        if isinstance(disps, (list, tuple)):
            disp = disps[0]  # 取最高分辨率的输出
        else:
            disp = disps
        return disp


def parse_args():
    parser = argparse.ArgumentParser(description='Export HDVO model to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='work_dirs/onnx_models',
        help='output directory for ONNX models')
    parser.add_argument(
        '--img-size',
        type=int,
        nargs=2,
        default=[256, 512],
        help='input image size (H W)')
    parser.add_argument(
        '--opset-version',
        type=int,
        default=11,
        help='ONNX opset version')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify ONNX model output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='simplify ONNX model using onnx-simplifier')
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='export with dynamic batch size')
    return parser.parse_args()


def export_depth_net(model, output_path, img_size, opset_version=11, 
                     verify=True, simplify=False, dynamic_batch=False):
    """Export depth network to ONNX"""
    H, W = img_size
    device = next(model.parameters()).device
    
    # 创建示例输入
    dummy_left = torch.randn(1, 3, H, W).to(device)
    dummy_right = torch.randn(1, 3, H, W).to(device)
    
    # 包装模型
    depth_wrapper = DepthNetWrapper(model.depth_net)
    depth_wrapper.eval()
    
    # 配置动态轴
    if dynamic_batch:
        dynamic_axes = {
            'left_img': {0: 'batch'},
            'right_img': {0: 'batch'},
            'disp': {0: 'batch'}
        }
    else:
        dynamic_axes = None
    
    print(f"Exporting depth network to {output_path}...")
    print(f"Input size: [B, 3, {H}, {W}]")
    
    # 导出ONNX
    torch.onnx.export(
        depth_wrapper,
        (dummy_left, dummy_right),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['left_img', 'right_img'],
        output_names=['disp'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"✓ ONNX model saved to {output_path}")
    
    # 简化模型
    if simplify:
        try:
            import onnxsim
            print("Simplifying ONNX model...")
            model_onnx = onnx.load(output_path)
            model_simp, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simp, output_path)
                print("✓ ONNX model simplified")
            else:
                print("⚠ Simplification check failed")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")
    
    # 验证模型
    if verify:
        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # 比较PyTorch和ONNX输出
        with torch.no_grad():
            torch_output = depth_wrapper(dummy_left, dummy_right)
        
        ort_session = ort.InferenceSession(output_path, 
                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        ort_inputs = {
            'left_img': dummy_left.cpu().numpy(),
            'right_img': dummy_right.cpu().numpy()
        }
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # 计算差异
        diff = np.abs(torch_output.cpu().numpy() - ort_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"✓ Output comparison:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✓ ONNX output matches PyTorch output!")
        else:
            print("⚠ Large difference detected, please check the model")
    
    return output_path


def benchmark_onnx(onnx_path, img_size, num_iterations=100, warmup=10):
    """Benchmark ONNX model inference speed"""
    H, W = img_size
    
    print(f"\nBenchmarking ONNX model: {onnx_path}")
    
    # 创建ONNX Runtime会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"Using provider: {ort_session.get_providers()[0]}")
    
    # 准备输入
    dummy_left = np.random.randn(1, 3, H, W).astype(np.float32)
    dummy_right = np.random.randn(1, 3, H, W).astype(np.float32)
    ort_inputs = {
        'left_img': dummy_left,
        'right_img': dummy_right
    }
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = ort_session.run(None, ort_inputs)
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    avg_time = times.mean()
    std_time = times.std()
    min_time = times.min()
    max_time = times.max()
    fps = 1.0 / avg_time
    
    print(f"\n{'='*50}")
    print(f"Benchmark Results (ONNX Runtime)")
    print(f"{'='*50}")
    print(f"Average time: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    print(f"Min time:     {min_time*1000:.2f} ms")
    print(f"Max time:     {max_time*1000:.2f} ms")
    print(f"Throughput:   {fps:.2f} FPS")
    print(f"{'='*50}\n")
    
    return avg_time


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建模型
    print("Building model...")
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    # 加载checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 移动到GPU并设置为评估模式
    model = model.cuda()
    model.eval()
    
    # 导出depth network
    depth_onnx_path = osp.join(args.output_dir, 'depth_net.onnx')
    export_depth_net(
        model, 
        depth_onnx_path, 
        args.img_size,
        opset_version=args.opset_version,
        verify=args.verify,
        simplify=args.simplify,
        dynamic_batch=args.dynamic_batch
    )
    
    # Benchmark
    benchmark_onnx(depth_onnx_path, args.img_size)
    
    print(f"\n{'='*50}")
    print("Export completed successfully!")
    print(f"ONNX model saved to: {depth_onnx_path}")
    print(f"{'='*50}")
    
    # 保存元数据
    metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'img_size': args.img_size,
        'opset_version': args.opset_version,
        'model_path': depth_onnx_path
    }
    
    import json
    metadata_path = osp.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
