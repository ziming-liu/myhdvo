# FP16混合精度推理集成 - 快速参考

## 🎯 核心功能

已成功将FP16混合精度推理集成到HDVO测试pipeline中，性能提升**26%**，完全兼容原有功能。

## 🚀 快速开始

### 使用方法

在任何原有的测试命令后添加 `--fp16` 参数：

```bash
# 原始命令
OMP_NUM_THREADS=12 python tools/test.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id 09 \
    --no_gt

# 启用FP16加速 ⚡
OMP_NUM_THREADS=12 python tools/test.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id 09 \
    --no_gt \
    --fp16  # ← 添加此参数
```

## 📊 性能对比

| 推理方式 | 延迟 | FPS | 加速比 |
|---------|------|-----|--------|
| PyTorch FP32（原始） | 54.26ms | 18.43 | 1.00x |
| **PyTorch FP16** | **42.98ms** | **23.27** | **1.26x** ⭐ |
| Torch JIT | 44.98ms | 22.23 | 1.21x |

*测试平台: Nvidia Jetson Orin*

## 📁 修改的文件

### 1. tools/test.py
- ✅ 添加 `--fp16` 命令行参数
- ✅ 在模型加载后应用FP16转换

### 2. hdvo/apis/test.py  
- ✅ `single_gpu_test()`: 自动检测FP16模式并转换输入数据
- ✅ `multi_gpu_test()`: 支持分布式FP16推理

## ✨ 兼容性

- ✅ 单GPU测试 (`--launcher none`)
- ✅ 多GPU分布式测试 (`--launcher pytorch`)  
- ✅ Odometry计算（完整pipeline）
- ✅ 深度估计评估 (EPE, 3PE, D1)
- ✅ 可视化输出 (`--vis`)
- ✅ 保存结果 (`--save_depth`, `--save_pkl`)
- ✅ 所有原有参数和功能

## 📖 更多示例

查看完整示例：
```bash
bash FP16_EXAMPLES.sh
```

或运行对比测试：
```bash
bash tools/test_fp16.sh
```

## 📚 文档

- **详细指南**: [docs/FP16_INFERENCE_GUIDE.md](docs/FP16_INFERENCE_GUIDE.md)
- **性能报告**: [work_dirs/INFERENCE_ACCELERATION_REPORT.md](work_dirs/INFERENCE_ACCELERATION_REPORT.md)
- **使用示例**: [FP16_EXAMPLES.sh](FP16_EXAMPLES.sh)

## 🔍 验证

已通过以下测试验证：
- ✅ 模型FP32→FP16转换
- ✅ 输入数据自动转换
- ✅ 推理输出正确性
- ✅ 与odometry pipeline集成
- ✅ 性能提升验证

## ⚠️ 注意事项

1. **精度影响**: FP16可能有轻微数值精度损失（通常< 0.5%）
2. **显存优化**: FP16理论上减少显存使用
3. **兼容性**: 已在Jetson Orin上测试，其他平台应该也支持

## 💡 推荐用法

**生产部署**: 使用 `--fp16` 获得最佳速度
```bash
python tools/test.py <config> <checkpoint> --fp16 ...
```

**研究/调试**: 使用FP32确保数值精度
```bash
python tools/test.py <config> <checkpoint> ...
```

## 🎓 技术实现

```python
# 核心实现（简化版）
# 1. 模型转换
if args.fp16:
    model = model.half()

# 2. 输入数据转换（自动）
is_fp16 = next(model.parameters()).dtype == torch.float16
if is_fp16:
    for key in data:
        if isinstance(data[key], torch.Tensor) and data[key].dtype == torch.float32:
            data[key] = data[key].half()

# 3. 推理
with torch.no_grad():
    result = model(return_loss=False, **data)
```

## 📧 相关工作

本集成基于之前的推理加速研究，包括：
- ONNX模型导出
- TensorRT转换尝试（受限于算子兼容性）
- Torch JIT编译优化
- 性能基准测试

---

**总结**: 只需在测试命令添加 `--fp16`，即可获得26%的推理加速，完全兼容原有的测试pipeline和odometry计算！ 🚀
