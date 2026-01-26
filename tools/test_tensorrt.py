"""
TensorRT Inference Script for HDVO Model
Author: ACENTAURI team
Date: 2025-01-27

使用TensorRT进行高性能推理
"""
import argparse
import os
import os.path as osp
import warnings
import numpy as np
import time
import cv2

import mmcv
import torch
from mmcv import Config

from hdvo.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT inference for HDVO')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file or TensorRT engine')
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
        '--img-size',
        type=int,
        nargs=2,
        default=[256, 512],
        help='input image size (H W)')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision for TensorRT')
    parser.add_argument(
        '--int8',
        action='store_true',
        help='Use INT8 precision for TensorRT (requires calibration)')
    parser.add_argument(
        '--engine-path',
        type=str,
        default=None,
        help='Path to save/load TensorRT engine')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference')
    return parser.parse_args()


try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Please install tensorrt and pycuda")


class TensorRTPredictor:
    """TensorRT深度预测器"""
    
    def __init__(self, engine_path, input_shape):
        """
        Args:
            engine_path: TensorRT engine文件路径
            input_shape: 输入形状 (B, C, H, W)
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        
        self.engine_path = engine_path
        self.input_shape = input_shape
        
        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载engine
        print(f"Loading TensorRT engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        # 统计信息
        self.timer = {
            'sum_time': 0.0,
            'count': 0,
            'avg_time': 0.0,
            'fps': 0.0
        }
        
        print(f"TensorRT engine loaded successfully")
        print(f"Number of inputs: {len(self.inputs)}")
        print(f"Number of outputs: {len(self.outputs)}")
    
    def predict(self, left_img, right_img):
        """
        预测深度图
        
        Args:
            left_img: [B, C, H, W] numpy array
            right_img: [B, C, H, W] numpy array
        
        Returns:
            disp: [B, 1, H, W] numpy array
        """
        # 转换为numpy
        if hasattr(left_img, 'cpu'):
            left_img = left_img.cpu().numpy()
        if hasattr(right_img, 'cpu'):
            right_img = right_img.cpu().numpy()
        
        left_img = left_img.astype(np.float32)
        right_img = right_img.astype(np.float32)
        
        # 拷贝输入数据到host
        np.copyto(self.inputs[0]['host'], left_img.ravel())
        np.copyto(self.inputs[1]['host'], right_img.ravel())
        
        # 开始计时
        start = time.time()
        
        # 将输入从host拷贝到device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 将输出从device拷贝到host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        # 同步
        self.stream.synchronize()
        
        end = time.time()
        
        # 更新统计信息
        self.timer['count'] += 1
        if self.timer['count'] > 5:
            self.timer['sum_time'] += (end - start)
            self.timer['avg_time'] = self.timer['sum_time'] / (self.timer['count'] - 5)
            self.timer['fps'] = 1.0 / self.timer['avg_time']
        
        # 获取输出
        output = self.outputs[0]['host']
        B, C, H, W = self.input_shape
        disp = output.reshape(B, 1, H, W)
        
        return disp
    
    def print_stats(self):
        """打印统计信息"""
        if self.timer['count'] > 5:
            print(f"\n{'='*60}")
            print(f"Inference Statistics (TensorRT)")
            print(f"{'='*60}")
            print(f"Average inference time: {self.timer['avg_time']*1000:.2f} ms")
            print(f"Average FPS: {self.timer['fps']:.2f}")
            print(f"Total frames processed: {self.timer['count']}")
            print(f"{'='*60}\n")
    
    def __del__(self):
        """清理资源"""
        del self.stream
        del self.context
        del self.engine


def build_tensorrt_engine(model, checkpoint_path, engine_path, img_size, 
                         batch_size=1, fp16=False, int8=False):
    """
    从PyTorch模型构建TensorRT engine
    
    Args:
        model: PyTorch模型
        checkpoint_path: checkpoint路径
        engine_path: engine保存路径
        img_size: 图像大小 (H, W)
        batch_size: batch大小
        fp16: 是否使用FP16
        int8: 是否使用INT8
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available")
    
    from hdvo.models import build_model
    from mmcv.runner import load_checkpoint
    
    print("Building TensorRT engine...")
    
    # 先导出ONNX
    onnx_path = engine_path.replace('.trt', '.onnx')
    
    # 使用torch2trt或直接从ONNX构建
    try:
        from torch2trt import torch2trt
        
        # 加载模型
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        model = model.cuda().eval()
        
        H, W = img_size
        left = torch.randn(batch_size, 3, H, W).cuda()
        right = torch.randn(batch_size, 3, H, W).cuda()
        
        # 构建TensorRT模型
        model_trt = torch2trt(
            model.depth_net,
            [left, right],
            fp16_mode=fp16,
            int8_mode=int8,
            max_workspace_size=1<<30,
            max_batch_size=batch_size
        )
        
        # 保存engine
        with open(engine_path, 'wb') as f:
            f.write(model_trt.engine.serialize())
        
        print(f"✓ TensorRT engine saved to {engine_path}")
        
    except ImportError:
        print("torch2trt not available, using ONNX -> TensorRT workflow")
        print("Please first export ONNX model using export_onnx.py")
        print("Then convert ONNX to TensorRT using:")
        print(f"  trtexec --onnx={onnx_path} --saveEngine={engine_path}")
        if fp16:
            print("         --fp16")
        if int8:
            print("         --int8")
        raise


def tensorrt_inference(predictor, data_loader):
    """
    使用TensorRT模型进行推理
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
        
        # 只处理第一帧
        ref_left = left_imgs[:, 0].numpy()
        ref_right = right_imgs[:, 0].numpy()
        
        # TensorRT推理
        disp = predictor.predict(ref_left, ref_right)
        
        # 转换为深度
        focal_val = data['focal'].numpy() if hasattr(data['focal'], 'numpy') else data['focal']
        baseline_val = data['baseline'].numpy() if hasattr(data['baseline'], 'numpy') else data['baseline']
        
        depth = (focal_val * baseline_val).reshape(-1, 1, 1, 1) / (disp + 1e-6)
        
        # 保存结果
        for b in range(B):
            pred_depths.append(depth[b, 0])
        
        # GT深度
        if 'left_depths' in data and len(data['left_depths']) > 0:
            gt_depth_batch = data['left_depths'].numpy()
            for b in range(B):
                gt_depths.append(gt_depth_batch[b, 0])
        
        batch_size = left_imgs.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    print()
    return pred_depths, gt_depths


def main():
    args = parse_args()
    
    if not TRT_AVAILABLE:
        print("Error: TensorRT is not available")
        print("Please install TensorRT and pycuda:")
        print("  pip install nvidia-tensorrt pycuda")
        return
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.data.videos_per_gpu = args.batch_size
    
    # 设置测试序列
    if args.test_seq_id is not None:
        cfg.test_seq_id = args.test_seq_id
        cfg.data.test.test_seq_id = args.test_seq_id
    
    test_seq_id = cfg.test_seq_id
    
    # 确定engine路径
    if args.engine_path is None:
        device_name = torch.cuda.get_device_name(0).replace(' ', '_').lower()
        precision = "fp16" if args.fp16 else ("int8" if args.int8 else "fp32")
        args.engine_path = f"work_dirs/depth_net_{precision}_{device_name}.trt"
    
    # 如果engine不存在，需要先构建
    if not osp.exists(args.engine_path):
        print(f"TensorRT engine not found at {args.engine_path}")
        print("Building engine from checkpoint...")
        
        from hdvo.models import build_model
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        
        build_tensorrt_engine(
            model,
            args.checkpoint,
            args.engine_path,
            args.img_size,
            batch_size=args.batch_size,
            fp16=args.fp16,
            int8=args.int8
        )
    
    # 创建预测器
    input_shape = (args.batch_size, 3, args.img_size[0], args.img_size[1])
    predictor = TensorRTPredictor(args.engine_path, input_shape)
    
    # 构建数据集
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    
    dataloader_setting = dict(
        videos_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False
    )
    data_loader = build_dataloader(dataset, **dataloader_setting)
    
    # 推理
    print(f"\nRunning TensorRT inference on sequence {test_seq_id}...")
    pred_depths, gt_depths = tensorrt_inference(predictor, data_loader)
    
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
            'tensorrt',
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
