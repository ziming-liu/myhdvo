"""
ONNX Inference Script for HDVO Model
Author: ACENTAURI team
Date: 2025-01-27

使用ONNX Runtime进行推理，支持GPU加速
"""
import argparse
import os
import os.path as osp
import warnings
import numpy as np
import time
import cv2

import mmcv
import onnxruntime as ort
from mmcv import Config

from hdvo.datasets import build_dataloader, build_dataset
from hdvo.core.evaluation import *


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX inference for HDVO')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('onnx_model', help='ONNX model file path')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g., EPE 3PE D1')
    parser.add_argument(
        '--test_seq_id',
        type=str,
        default="09",
        help="test sequence ID")
    parser.add_argument(
        '--max_depth',
        type=int,
        default=655,
        help="max depth value")
    parser.add_argument(
        '--save_depth',
        action='store_true',
        help="save depth predictions")
    parser.add_argument(
        '--provider',
        type=str,
        default='cuda',
        choices=['cuda', 'tensorrt', 'cpu'],
        help='ONNX Runtime execution provider')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 inference (TensorRT only)')
    return parser.parse_args()


class ONNXDepthPredictor:
    """ONNX Runtime深度预测器"""
    
    def __init__(self, onnx_path, provider='cuda', fp16=False):
        """
        Args:
            onnx_path: ONNX模型路径
            provider: 'cuda', 'tensorrt', or 'cpu'
            fp16: 是否使用FP16推理（仅TensorRT支持）
        """
        self.onnx_path = onnx_path
        self.fp16 = fp16
        
        # 设置执行提供者
        if provider == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        elif provider == 'tensorrt':
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                'trt_fp16_enable': fp16,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'work_dirs/trt_cache',
            }
            providers = [
                ('TensorrtExecutionProvider', trt_options),
                ('CUDAExecutionProvider', {}),
                'CPUExecutionProvider'
            ]
            os.makedirs('work_dirs/trt_cache', exist_ok=True)
        else:  # cpu
            providers = ['CPUExecutionProvider']
        
        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"Loading ONNX model from {onnx_path}")
        print(f"Using provider: {provider}")
        if fp16:
            print("FP16 inference enabled")
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # 打印实际使用的provider
        print(f"Active providers: {self.session.get_providers()}")
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
        
        # 统计信息
        self.timer = {
            'sum_time': 0.0,
            'count': 0,
            'avg_time': 0.0,
            'fps': 0.0
        }
    
    def predict(self, left_img, right_img):
        """
        预测深度图
        
        Args:
            left_img: [B, 3, H, W] numpy array or torch tensor
            right_img: [B, 3, H, W] numpy array or torch tensor
        
        Returns:
            disp: [B, 1, H, W] numpy array
        """
        # 转换为numpy
        if hasattr(left_img, 'cpu'):
            left_img = left_img.cpu().numpy()
        if hasattr(right_img, 'cpu'):
            right_img = right_img.cpu().numpy()
        
        # 确保是float32
        left_img = left_img.astype(np.float32)
        right_img = right_img.astype(np.float32)
        
        # 推理
        start = time.time()
        ort_inputs = {
            self.input_names[0]: left_img,
            self.input_names[1]: right_img
        }
        disp = self.session.run(self.output_names, ort_inputs)[0]
        end = time.time()
        
        # 更新统计信息
        self.timer['count'] += 1
        if self.timer['count'] > 5:  # 跳过前5次warmup
            self.timer['sum_time'] += (end - start)
            self.timer['avg_time'] = self.timer['sum_time'] / (self.timer['count'] - 5)
            self.timer['fps'] = 1.0 / self.timer['avg_time']
        
        return disp
    
    def print_stats(self):
        """打印统计信息"""
        if self.timer['count'] > 5:
            print(f"\n{'='*60}")
            print(f"Inference Statistics (ONNX Runtime)")
            print(f"{'='*60}")
            print(f"Average inference time: {self.timer['avg_time']*1000:.2f} ms")
            print(f"Average FPS: {self.timer['fps']:.2f}")
            print(f"Total frames processed: {self.timer['count']}")
            print(f"{'='*60}\n")


def onnx_inference(predictor, data_loader, focal, baseline):
    """
    使用ONNX模型进行推理
    
    Args:
        predictor: ONNXDepthPredictor实例
        data_loader: 数据加载器
        focal: 焦距
        baseline: 基线
    
    Returns:
        pred_depths: 预测深度列表
        gt_depths: GT深度列表
    """
    pred_depths = []
    gt_depths = []
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        # 获取左右图像
        left_imgs = data['left_imgs']  # [B, T, C, H, W]
        right_imgs = data['right_imgs']  # [B, T, C, H, W]
        
        B, T, C, H, W = left_imgs.shape
        
        # 只处理第一帧（参考帧）
        ref_left = left_imgs[:, 0].numpy()  # [B, C, H, W]
        ref_right = right_imgs[:, 0].numpy()  # [B, C, H, W]
        
        # ONNX推理
        disp = predictor.predict(ref_left, ref_right)  # [B, 1, H, W]
        
        # 转换为深度
        focal_val = data['focal'].numpy() if hasattr(data['focal'], 'numpy') else data['focal']
        baseline_val = data['baseline'].numpy() if hasattr(data['baseline'], 'numpy') else data['baseline']
        
        depth = (focal_val * baseline_val).reshape(-1, 1, 1, 1) / (disp + 1e-6)
        
        # 保存结果
        for b in range(B):
            pred_depths.append(depth[b, 0])  # [H, W]
        
        # 获取GT深度（如果有）
        if 'left_depths' in data and len(data['left_depths']) > 0:
            gt_depth_batch = data['left_depths'].numpy()  # [B, T, H, W]
            for b in range(B):
                gt_depths.append(gt_depth_batch[b, 0])  # [H, W]
        
        # 更新进度条
        batch_size = left_imgs.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    print()  # 换行
    
    return pred_depths, gt_depths


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.data.videos_per_gpu = 1
    
    # 设置测试序列
    if args.test_seq_id is not None:
        cfg.test_seq_id = args.test_seq_id
        cfg.data.test.test_seq_id = args.test_seq_id
    
    test_seq_id = cfg.test_seq_id
    
    # 构建数据集
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    
    dataloader_setting = dict(
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False
    )
    data_loader = build_dataloader(dataset, **dataloader_setting)
    
    # 创建ONNX预测器
    predictor = ONNXDepthPredictor(
        args.onnx_model,
        provider=args.provider,
        fp16=args.fp16
    )
    
    # 推理
    print(f"\nRunning inference on sequence {test_seq_id}...")
    pred_depths, gt_depths = onnx_inference(
        predictor,
        data_loader,
        focal=cfg.data.test.get('focal', 360.0),
        baseline=cfg.data.test.get('baseline', 0.54)
    )
    
    # 打印统计信息
    predictor.print_stats()
    
    # 后处理
    pred_depths_np = []
    for depth in pred_depths:
        depth_np = depth if isinstance(depth, np.ndarray) else depth
        depth_np[depth_np > args.max_depth] = args.max_depth
        depth_np[np.isnan(depth_np)] = args.max_depth
        depth_np[np.isinf(depth_np)] = args.max_depth
        pred_depths_np.append(depth_np)
    
    # 保存深度图
    if args.save_depth:
        from tools.outputs_proc.save_load_depth import save_depth_maps
        print(f"\nSaving depth maps to {cfg.work_dir}...")
        save_depth_maps(
            cfg.dataset_type,
            test_seq_id,
            pred_depths_np,
            cfg.work_dir,
            'onnx',
            stereo_view="left",
            min_depth=1,
            max_depth=args.max_depth,
            first_frame_id=0
        )
        print("✓ Depth maps saved")
    
    # 评估
    if args.eval and len(gt_depths) > 0:
        print(f"\nEvaluating on sequence {test_seq_id}...")
        results = dataset.evaluate(
            [(pred_depths_np, gt_depths)],
            metrics=args.eval
        )
        print("\nEvaluation Results:")
        print("="*60)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("="*60)


if __name__ == '__main__':
    main()
