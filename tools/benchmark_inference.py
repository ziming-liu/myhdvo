"""
快速推理性能测试脚本
对比PyTorch原生推理、torch.jit优化、CUDA优化等方法
"""
import argparse
import time
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from hdvo.models import build_model

def benchmark_pytorch(model, left, right, warmup=10, iterations=100):
    """PyTorch原生推理"""
    print("\n" + "="*60)
    print("测试 PyTorch 原生推理")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model.depth_net(left, right)
        
        # Benchmark
        torch.cuda.synchronize()
        times = []
        for i in range(iterations):
            start = time.time()
            _ = model.depth_net(left, right)
            torch.cuda.synchronize()
            times.append(time.time() - start)
            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{iterations}")
        
        times = np.array(times)
        print(f"\n平均推理时间: {times.mean()*1000:.2f} ms (±{times.std()*1000:.2f} ms)")
        print(f"最小时间: {times.min()*1000:.2f} ms")
        print(f"最大时间: {times.max()*1000:.2f} ms")
        print(f"FPS: {1.0/times.mean():.2f}")
    
    return times.mean()

def benchmark_jit(model, left, right, warmup=10, iterations=100):
    """Torch JIT优化推理"""
    print("\n" + "="*60)
    print("测试 Torch JIT 脚本化推理")
    print("="*60)
    
    try:
        # 尝试JIT脚本化
        print("正在进行JIT脚本化...")
        model_jit = torch.jit.script(model.depth_net)
        print("JIT脚本化成功")
    except Exception as e:
        print(f"JIT脚本化失败: {e}")
        print("尝试JIT追踪...")
        try:
            model_jit = torch.jit.trace(model.depth_net, (left, right))
            print("JIT追踪成功")
        except Exception as e2:
            print(f"JIT追踪也失败: {e2}")
            return None
    
    model_jit.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model_jit(left, right)
        
        # Benchmark
        torch.cuda.synchronize()
        times = []
        for i in range(iterations):
            start = time.time()
            _ = model_jit(left, right)
            torch.cuda.synchronize()
            times.append(time.time() - start)
            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{iterations}")
        
        times = np.array(times)
        print(f"\n平均推理时间: {times.mean()*1000:.2f} ms (±{times.std()*1000:.2f} ms)")
        print(f"最小时间: {times.min()*1000:.2f} ms")
        print(f"最大时间: {times.max()*1000:.2f} ms")
        print(f"FPS: {1.0/times.mean():.2f}")
    
    return times.mean()

def benchmark_fp16(model, left, right, warmup=10, iterations=100):
    """FP16混合精度推理"""
    print("\n" + "="*60)
    print("测试 FP16 混合精度推理")
    print("="*60)
    
    model.half()
    left_fp16 = left.half()
    right_fp16 = right.half()
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model.depth_net(left_fp16, right_fp16)
        
        # Benchmark
        torch.cuda.synchronize()
        times = []
        for i in range(iterations):
            start = time.time()
            _ = model.depth_net(left_fp16, right_fp16)
            torch.cuda.synchronize()
            times.append(time.time() - start)
            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{iterations}")
        
        times = np.array(times)
        print(f"\n平均推理时间: {times.mean()*1000:.2f} ms (±{times.std()*1000:.2f} ms)")
        print(f"最小时间: {times.min()*1000:.2f} ms")
        print(f"最大时间: {times.max()*1000:.2f} ms")
        print(f"FPS: {1.0/times.mean():.2f}")
    
    return times.mean()

def main():
    parser = argparse.ArgumentParser(description='推理性能测试')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='checkpoint文件路径')
    parser.add_argument('--img-size', type=int, nargs=2, default=[320, 1024], help='输入图像尺寸 (H W)')
    parser.add_argument('--warmup', type=int, default=10, help='预热迭代次数')
    parser.add_argument('--iterations', type=int, default=100, help='测试迭代次数')
    args = parser.parse_args()
    
    print("="*60)
    print("HDVO 推理性能测试")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"输入尺寸: {args.img_size[0]}x{args.img_size[1]}")
    print(f"预热次数: {args.warmup}")
    print(f"测试次数: {args.iterations}")
    
    # 加载配置和模型
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda()
    
    # 准备输入
    H, W = args.img_size
    left = torch.randn(1, 3, H, W).cuda()
    right = torch.randn(1, 3, H, W).cuda()
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 运行测试
    results = {}
    
    # 1. PyTorch原生
    results['PyTorch'] = benchmark_pytorch(model, left, right, args.warmup, args.iterations)
    
    # 2. Torch JIT
    jit_time = benchmark_jit(model, left, right, args.warmup, args.iterations)
    if jit_time is not None:
        results['Torch JIT'] = jit_time
    
    # 3. FP16
    try:
        # 重新加载模型用于FP16测试
        model_fp16 = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model_fp16, args.checkpoint, map_location='cpu')
        model_fp16 = model_fp16.cuda()
        results['FP16'] = benchmark_fp16(model_fp16, left, right, args.warmup, args.iterations)
    except Exception as e:
        print(f"\nFP16测试失败: {e}")
    
    # 总结
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    print(f"{'方法':<15} {'平均时间(ms)':<15} {'FPS':<10} {'加速比':<10}")
    print("-"*60)
    
    baseline = results['PyTorch']
    for method, avg_time in results.items():
        fps = 1.0 / avg_time
        speedup = baseline / avg_time
        print(f"{method:<15} {avg_time*1000:<15.2f} {fps:<10.2f} {speedup:<10.2f}x")
    
    print("="*60)
    
    # 保存结果
    import json
    result_file = 'work_dirs/benchmark_results.json'
    with open(result_file, 'w') as f:
        json.dump({
            'config': args.config,
            'checkpoint': args.checkpoint,
            'img_size': args.img_size,
            'gpu': torch.cuda.get_device_name(0),
            'results': {k: {'time_ms': v*1000, 'fps': 1.0/v} for k, v in results.items()}
        }, f, indent=2)
    print(f"\n结果已保存到: {result_file}")

if __name__ == '__main__':
    main()
