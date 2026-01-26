# FP16 混合精度推理加速指南

## 概述

FP16（半精度浮点）推理已集成到HDVO测试pipeline中，可以在保持精度的同时提升约**26%的推理速度**。

## 快速开始

### 基本用法

在原有的测试命令后添加 `--fp16` 参数即可启用FP16加速：

```bash
# 原始FP32推理
OMP_NUM_THREADS=12 python tools/test.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id 09 \
    --no_gt

# 启用FP16加速推理
OMP_NUM_THREADS=12 python tools/test.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id 09 \
    --no_gt \
    --fp16  # 添加此参数
```

### 使用测试脚本

我们提供了一个便捷的测试脚本，可以对比FP32和FP16的性能：

```bash
bash tools/test_fp16.sh
```

## 性能对比

基于Nvidia Jetson Orin平台的测试结果：

| 推理方式 | 延迟 | FPS | 加速比 |
|---------|------|-----|--------|
| PyTorch FP32 | 54.26ms | 18.43 | 1.00x |
| PyTorch FP16 | 42.98ms | 23.27 | **1.26x** |

## 技术细节

### 实现原理

FP16推理通过以下方式实现：

1. **模型转换**: 将整个模型的参数和缓冲区转换为FP16格式
   ```python
   model = model.half()
   ```

2. **输入数据转换**: 自动检测模型精度并转换输入数据
   ```python
   if is_fp16:
       for key in data:
           if isinstance(data[key], torch.Tensor) and data[key].dtype == torch.float32:
               data[key] = data[key].half()
   ```

3. **输出转换**: 输出结果自动转换回FP32以保持兼容性

### 修改的文件

1. **tools/test.py**
   - 添加 `--fp16` 命令行参数
   - 在模型加载后应用FP16转换

2. **hdvo/apis/test.py**
   - `single_gpu_test()`: 支持FP16数据转换
   - `multi_gpu_test()`: 支持分布式FP16推理

### 兼容性

- ✅ 单GPU测试 (`--launcher none`)
- ✅ 多GPU分布式测试 (`--launcher pytorch`)
- ✅ Odometry计算
- ✅ 深度估计评估 (EPE, 3PE, D1)
- ✅ 可视化输出
- ✅ 保存预测结果 (`--save_depth`, `--save_pkl`)

## 注意事项

### 精度影响

FP16推理可能会有轻微的数值精度损失，但在大多数情况下：
- 视觉效果无明显差异
- 评估指标变化 < 0.5%
- 可以通过在验证集上测试来验证精度

### 何时使用FP16

**推荐使用场景**:
- 生产部署，需要实时推理
- 边缘设备（Jetson系列）
- 批量处理大量数据
- GPU内存受限的情况

**不推荐使用场景**:
- 需要极高数值精度的研究
- 模型训练（训练应使用混合精度训练而非纯FP16）
- 调试阶段（建议先用FP32确保正确性）

## 进一步优化

如果需要更高的推理速度，可以考虑：

1. **Torch JIT编译** (1.21x加速)
   ```bash
   python tools/test.py ... --fuse-conv-bn
   ```

2. **批处理** (增加batch size)
   - 修改配置文件中的 `samples_per_gpu`

3. **TensorRT** (预期2-3x加速)
   - 需要解决ONNX算子兼容性问题
   - 参考 `work_dirs/INFERENCE_ACCELERATION_REPORT.md`

## 故障排除

### 问题: CUDA out of memory

**解决方案**: FP16理论上会降低显存使用，但如果仍有问题：
```bash
# 减小batch size
python tools/test.py ... --cfg-options data.test.samples_per_gpu=1
```

### 问题: 数值不稳定或NaN

**解决方案**: 某些操作在FP16下可能不稳定，可以：
1. 检查模型中是否有除法或指数操作
2. 考虑使用混合精度（部分层保持FP32）
3. 如果问题持续，使用FP32推理

### 问题: 性能提升不明显

**可能原因**:
- 数据加载成为瓶颈（增加 `num_workers`）
- CPU计算占比高（检查是否有CPU-only操作）
- GPU未充分利用（考虑增加batch size）

## 示例输出

```
## Converting model to FP16 for faster inference
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 1591/1591, 23.3 task/s, elapsed: 68s
Average inference time: 42.98 ms
FPS: 23.27
```

## 相关资源

- 详细性能测试报告: `work_dirs/INFERENCE_ACCELERATION_REPORT.md`
- ONNX导出脚本: `tools/export_onnx.py`
- 性能基准测试: `tools/benchmark_inference.py`
- 快速推理指南: `INFERENCE_QUICKSTART.md`

## 参考文献

1. [Mixed Precision Training - NVIDIA](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
2. [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
3. [MMCV FP16 Utils](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/fp16_utils.py)
