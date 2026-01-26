#!/usr/bin/env python3
"""
ONNX GPU推理性能测试脚本
使用ONNX Runtime的CUDA/TensorRT执行提供器进行加速推理
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path


def benchmark_onnx_inference(onnx_path, num_iterations=100, provider='CUDAExecutionProvider', use_fp16=False):
    """
    对ONNX模型进行性能基准测试
    
    Args:
        onnx_path: ONNX模型路径
        num_iterations: 测试迭代次数
        provider: 执行提供器 (CUDAExecutionProvider, TensorrtExecutionProvider, CPUExecutionProvider)
        use_fp16: 是否使用FP16精度
    """
    print(f"\n{'='*60}")
    print(f"ONNX Runtime GPU 推理性能测试")
    print(f"{'='*60}")
    print(f"模型路径: {onnx_path}")
    print(f"执行提供器: {provider}")
    print(f"迭代次数: {num_iterations}")
    print(f"使用FP16: {use_fp16}")
    print(f"{'='*60}\n")
    
    # 检查提供器是否可用
    available_providers = ort.get_available_providers()
    print(f"可用的提供器: {available_providers}\n")
    
    if provider not in available_providers:
        print(f"警告: {provider} 不可用，回退到 CPUExecutionProvider")
        provider = 'CPUExecutionProvider'
    
    # 创建推理会话
    print("正在加载ONNX模型...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # TensorRT特定选项
    if provider == 'TensorrtExecutionProvider':
        trt_options = {
            'trt_fp16_enable': use_fp16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(Path(onnx_path).parent / 'trt_cache'),
        }
        providers = [(provider, trt_options), 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = [provider, 'CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # 获取输入输出信息
    input_name_left = session.get_inputs()[0].name
    input_name_right = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"✓ 模型加载成功")
    print(f"  输入名称: {input_name_left}, {input_name_right}")
    print(f"  输入形状: {input_shape}")
    print(f"  输出名称: {output_name}")
    print(f"  实际使用的提供器: {session.get_providers()}\n")
    
    # 准备测试数据
    dtype = np.float16 if use_fp16 else np.float32
    batch_size, channels, height, width = input_shape
    left_img = np.random.randn(batch_size, channels, height, width).astype(dtype)
    right_img = np.random.randn(batch_size, channels, height, width).astype(dtype)
    
    # 预热
    print("预热中...")
    for _ in range(10):
        _ = session.run([output_name], {
            input_name_left: left_img,
            input_name_right: right_img
        })
    
    # 性能测试
    print(f"\n开始性能测试 ({num_iterations} 次迭代)...")
    times = []
    
    for i in range(num_iterations):
        start = time.time()
        output = session.run([output_name], {
            input_name_left: left_img,
            input_name_right: right_img
        })
        end = time.time()
        times.append((end - start) * 1000)  # 转换为毫秒
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_iterations}")
    
    # 统计结果
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000.0 / avg_time
    
    print(f"\n{'='*60}")
    print(f"性能测试结果 ({provider})")
    print(f"{'='*60}")
    print(f"平均推理时间: {avg_time:.2f} ms (± {std_time:.2f} ms)")
    print(f"最小推理时间: {min_time:.2f} ms")
    print(f"最大推理时间: {max_time:.2f} ms")
    print(f"平均FPS: {fps:.2f}")
    print(f"吞吐量: {fps:.2f} 帧/秒")
    print(f"{'='*60}\n")
    
    return {
        'provider': provider,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'use_fp16': use_fp16
    }


def main():
    parser = argparse.ArgumentParser(description='ONNX GPU推理性能测试')
    parser.add_argument(
        '--onnx-path',
        type=str,
        default='work_dirs/onnx_models/depth_net.onnx',
        help='ONNX模型路径'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='测试迭代次数'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='CUDAExecutionProvider',
        choices=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'],
        help='执行提供器'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用FP16精度（仅TensorRT）'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='对比所有可用的提供器'
    )
    
    args = parser.parse_args()
    
    if args.compare_all:
        # 对比测试所有提供器
        print("\n" + "="*60)
        print("对比测试所有提供器")
        print("="*60 + "\n")
        
        results = []
        providers_to_test = []
        
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers_to_test.append(('CUDAExecutionProvider', False))
        if 'TensorrtExecutionProvider' in available:
            providers_to_test.append(('TensorrtExecutionProvider', False))
            providers_to_test.append(('TensorrtExecutionProvider', True))  # FP16
        providers_to_test.append(('CPUExecutionProvider', False))
        
        for provider, use_fp16 in providers_to_test:
            result = benchmark_onnx_inference(
                args.onnx_path,
                args.iterations,
                provider,
                use_fp16
            )
            results.append(result)
        
        # 打印对比表格
        print("\n" + "="*80)
        print("性能对比总结")
        print("="*80)
        print(f"{'提供器':<30} {'精度':<8} {'平均时间':<12} {'FPS':<10} {'加速比':<10}")
        print("-"*80)
        
        baseline_time = results[0]['avg_time_ms'] if results else 1.0
        for r in results:
            provider_name = r['provider'].replace('ExecutionProvider', '')
            precision = 'FP16' if r['use_fp16'] else 'FP32'
            speedup = baseline_time / r['avg_time_ms']
            print(f"{provider_name:<30} {precision:<8} {r['avg_time_ms']:>8.2f} ms  {r['fps']:>7.2f}   {speedup:>6.2f}x")
        
        print("="*80 + "\n")
        
    else:
        # 单个提供器测试
        benchmark_onnx_inference(
            args.onnx_path,
            args.iterations,
            args.provider,
            args.fp16
        )


if __name__ == '__main__':
    main()
