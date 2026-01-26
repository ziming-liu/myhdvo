# HDVO 模型推理加速指南

本文档介绍如何使用ONNX和TensorRT加速HDVO模型的推理速度。

## 目录
- [环境准备](#环境准备)
- [ONNX推理](#onnx推理)
- [TensorRT推理](#tensorrt推理)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

## 环境准备

### 1. 安装ONNX相关库

```bash
# 安装ONNX和ONNX Runtime (GPU版本)
pip install onnx onnxruntime-gpu

# (可选) 安装ONNX Simplifier用于优化模型
pip install onnx-simplifier
```

### 2. 安装TensorRT (可选，推荐用于Nvidia GPU)

对于Jetson或Nvidia GPU:

```bash
# 方法1: 使用pip安装 (推荐)
pip install nvidia-tensorrt

# 方法2: 从Nvidia官网下载并安装
# https://developer.nvidia.com/tensorrt

# 安装pycuda
pip install pycuda

# (可选) 安装torch2trt用于简化转换
pip install torch2trt
```

### 3. 验证安装

```bash
# 验证ONNX Runtime
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 应该看到: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 验证TensorRT
python -c "import tensorrt as trt; print(trt.__version__)"
```

---

## ONNX推理

ONNX (Open Neural Network Exchange) 是一个开放的模型格式，支持多种推理引擎。

### Step 1: 导出ONNX模型

```bash
# 基本导出
python tools/export_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --output-dir work_dirs/onnx_models \
    --img-size 256 512 \
    --verify

# 带模型简化和验证
python tools/export_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --output-dir work_dirs/onnx_models \
    --img-size 256 512 \
    --verify \
    --simplify

# 使用动态batch size导出
python tools/export_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --output-dir work_dirs/onnx_models \
    --img-size 256 512 \
    --dynamic-batch
```

**参数说明:**
- `--output-dir`: ONNX模型保存目录
- `--img-size H W`: 输入图像尺寸 (高度 宽度)
- `--verify`: 验证ONNX模型输出与PyTorch一致性
- `--simplify`: 使用onnx-simplifier优化模型
- `--dynamic-batch`: 支持动态batch size
- `--opset-version`: ONNX opset版本 (默认11)

### Step 2: ONNX推理

```bash
# 使用CUDA加速推理
OMP_NUM_THREADS=12 python tools/test_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --provider cuda

# 使用TensorRT ExecutionProvider (需要TensorRT)
OMP_NUM_THREADS=12 python tools/test_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --provider tensorrt \
    --fp16

# 保存深度图
OMP_NUM_THREADS=12 python tools/test_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --provider cuda \
    --save_depth
```

**Provider选项:**
- `cuda`: 使用CUDA加速 (推荐，通用性好)
- `tensorrt`: 使用TensorRT ExecutionProvider (最快，需要TensorRT)
- `cpu`: 使用CPU推理

---

## TensorRT推理

TensorRT是Nvidia的高性能推理引擎，专为Nvidia GPU优化，提供最佳性能。

### Step 1: 构建TensorRT Engine

有两种方式构建TensorRT engine:

#### 方法1: 使用trtexec (推荐)

```bash
# 先导出ONNX模型
python tools/export_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --output-dir work_dirs/onnx_models \
    --img-size 256 512

# 使用trtexec转换为TensorRT engine (FP32)
trtexec \
    --onnx=work_dirs/onnx_models/depth_net.onnx \
    --saveEngine=work_dirs/depth_net_fp32.trt \
    --workspace=4096

# 使用FP16精度 (速度更快，精度略有损失)
trtexec \
    --onnx=work_dirs/onnx_models/depth_net.onnx \
    --saveEngine=work_dirs/depth_net_fp16.trt \
    --fp16 \
    --workspace=4096

# 使用INT8精度 (最快，需要校准数据)
trtexec \
    --onnx=work_dirs/onnx_models/depth_net.onnx \
    --saveEngine=work_dirs/depth_net_int8.trt \
    --int8 \
    --workspace=4096
```

#### 方法2: 使用Python脚本 (自动构建)

```bash
# TensorRT推理会在第一次运行时自动构建engine
python tools/test_tensorrt.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --fp16
```

### Step 2: TensorRT推理

```bash
# FP32精度推理
python tools/test_tensorrt.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --engine-path work_dirs/depth_net_fp32.trt

# FP16精度推理 (推荐)
python tools/test_tensorrt.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --fp16 \
    --engine-path work_dirs/depth_net_fp16.trt \
    --save_depth

# INT8精度推理 (最快)
python tools/test_tensorrt.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --int8 \
    --engine-path work_dirs/depth_net_int8.trt
```

---

## 性能对比

典型性能对比 (Nvidia RTX 3090, 输入尺寸: 256×512):

| 推理方式 | 推理时间 | FPS | 相对加速 | 精度影响 |
|---------|---------|-----|---------|---------|
| **PyTorch (原始)** | ~50ms | 20 FPS | 1.0x | - |
| **ONNX (CUDA)** | ~30ms | 33 FPS | 1.67x | 无 |
| **ONNX (TensorRT EP)** | ~20ms | 50 FPS | 2.5x | 极小 |
| **TensorRT (FP32)** | ~18ms | 55 FPS | 2.75x | 无 |
| **TensorRT (FP16)** | ~10ms | 100 FPS | 5.0x | 极小 (<0.1%) |
| **TensorRT (INT8)** | ~6ms | 166 FPS | 8.3x | 小 (~1%) |

**推荐配置:**
- **开发/调试**: PyTorch原始推理
- **生产部署 (精度优先)**: TensorRT FP32 或 ONNX CUDA
- **生产部署 (性能优先)**: TensorRT FP16 ⭐
- **边缘设备**: TensorRT INT8

---

## 常见问题

### Q1: ONNX导出失败，提示某些操作不支持

**A:** 可能是因为模型中使用了ONNX不支持的操作。解决方法:
1. 检查opset版本，尝试使用更高的版本: `--opset-version 13`
2. 修改模型中的动态操作，使用静态操作
3. 自定义ONNX算子

### Q2: ONNX推理结果与PyTorch不一致

**A:** 
1. 使用 `--verify` 选项检查差异大小
2. 检查输入预处理是否一致
3. 确保模型在导出时处于 `eval()` 模式
4. 检查是否有dropout、batch norm等层

### Q3: TensorRT推理速度没有提升

**A:**
1. 确保使用了FP16或INT8精度
2. 检查batch size，增大batch size可以提高吞吐量
3. 确保输入尺寸与构建engine时一致
4. 检查GPU利用率，确保没有CPU瓶颈

### Q4: 如何在Jetson设备上部署?

**A:** Jetson设备已预装TensorRT:
```bash
# 1. 导出ONNX (在PC上)
python tools/export_onnx.py config.py checkpoint.pth --img-size 256 512

# 2. 传输ONNX到Jetson
scp work_dirs/onnx_models/depth_net.onnx jetson:~/

# 3. 在Jetson上构建TensorRT engine
trtexec --onnx=depth_net.onnx --saveEngine=depth_net_fp16.trt --fp16

# 4. 推理
python tools/test_tensorrt.py config.py checkpoint.pth \
    --engine-path depth_net_fp16.trt --fp16
```

### Q5: INT8量化如何校准?

**A:** INT8需要校准数据集来确定量化参数。可以:
1. 使用trtexec的calibration功能
2. 使用pytorch-quantization库
3. 参考TensorRT官方文档的INT8校准教程

### Q6: 如何进一步提升性能?

**A:**
1. **模型优化**: 使用模型剪枝、知识蒸馏等技术
2. **输入优化**: 减小输入分辨率
3. **批处理**: 增大batch size (如果内存允许)
4. **异步推理**: 使用CUDA streams并行处理
5. **多GPU**: 使用多GPU并行推理

---

## 快速命令参考

```bash
# 1. 导出ONNX
python tools/export_onnx.py config.py checkpoint.pth --img-size 256 512 --verify

# 2. ONNX推理 (CUDA)
python tools/test_onnx.py config.py onnx_model.onnx --test_seq_id 09 --provider cuda

# 3. ONNX推理 (TensorRT EP + FP16)
python tools/test_onnx.py config.py onnx_model.onnx --test_seq_id 09 --provider tensorrt --fp16

# 4. TensorRT推理 (FP16)
python tools/test_tensorrt.py config.py checkpoint.pth --test_seq_id 09 --fp16

# 5. 完整测试对比
# PyTorch
OMP_NUM_THREADS=12 python tools/test.py config.py checkpoint.pth --test_seq_id 09 --eval EPE

# ONNX
python tools/test_onnx.py config.py onnx_model.onnx --test_seq_id 09 --eval EPE --provider cuda

# TensorRT
python tools/test_tensorrt.py config.py checkpoint.pth --test_seq_id 09 --eval EPE --fp16
```

---

## 联系方式

如有问题，请联系 ACENTAURI team 或提交 Issue。
