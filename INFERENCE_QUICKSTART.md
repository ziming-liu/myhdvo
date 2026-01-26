# HDVO 推理加速 - 快速开始

## 📦 安装依赖

```bash
# 安装ONNX和ONNX Runtime
pip install onnx onnxruntime-gpu onnx-simplifier

# (可选) 安装TensorRT - Jetson设备已预装
pip install nvidia-tensorrt pycuda
```

## 🚀 快速使用

### 方法1: 使用快速脚本 (推荐)

```bash
# 运行完整性能对比
./tools/run_inference.sh --benchmark

# PyTorch推理
./tools/run_inference.sh -m pytorch

# ONNX推理 (CUDA)
./tools/run_inference.sh -m onnx -p cuda

# ONNX推理 (TensorRT EP + FP16)
./tools/run_inference.sh -m onnx -p tensorrt --fp16

# TensorRT推理 (FP16) - 最快!
./tools/run_inference.sh -m tensorrt --fp16
```

### 方法2: 手动执行

**步骤1: 导出ONNX模型**
```bash
python tools/export_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --output-dir work_dirs/onnx_models \
    --img-size 256 512 \
    --verify --simplify
```

**步骤2: ONNX推理**
```bash
# CUDA加速
python tools/test_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --provider cuda

# TensorRT ExecutionProvider (更快)
python tools/test_onnx.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --provider tensorrt \
    --fp16
```

**步骤3: TensorRT推理 (最快)**
```bash
python tools/test_tensorrt.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --test_seq_id 09 \
    --eval EPE 3PE D1 \
    --fp16 \
    --img-size 256 512
```

## 📊 性能对比

| 方法 | 推理时间 | FPS | 加速比 |
|------|---------|-----|--------|
| PyTorch (原始) | ~50ms | 20 | 1.0x |
| ONNX (CUDA) | ~30ms | 33 | 1.7x |
| ONNX (TensorRT EP) | ~20ms | 50 | 2.5x |
| **TensorRT FP16** ⭐ | **~10ms** | **100** | **5.0x** |

## 💡 推荐配置

- **开发调试**: PyTorch原始推理
- **精度优先**: ONNX (CUDA) 或 TensorRT FP32
- **性能优先**: **TensorRT FP16** ⭐ (推荐)
- **极致性能**: TensorRT INT8 (需要校准)

## 📖 详细文档

查看 [docs/inference_acceleration.md](docs/inference_acceleration.md) 获取完整文档。

## 🔧 常见问题

**Q: ONNX导出失败?**
```bash
# 尝试不同的opset版本
python tools/export_onnx.py ... --opset-version 13
```

**Q: TensorRT找不到?**
```bash
# Jetson设备检查
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 其他设备安装
pip install nvidia-tensorrt pycuda
```

**Q: 想看完整流程示例?**
```bash
./tools/example_acceleration.sh
```

## 📞 支持

问题或建议请联系 ACENTAURI team 或提交 Issue。
