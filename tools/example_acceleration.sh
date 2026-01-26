#!/bin/bash
# 快速示例: HDVO推理加速完整流程
# 从PyTorch模型到ONNX/TensorRT加速推理

echo "=========================================="
echo "HDVO 推理加速 - 快速示例"
echo "=========================================="

# 配置
CONFIG="configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py"
CHECKPOINT="work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth"
SEQ_ID="09"

echo ""
echo "步骤1: 原始PyTorch推理 (基线性能)"
echo "----------------------------------------"
OMP_NUM_THREADS=12 python tools/test.py \
    $CONFIG $CHECKPOINT \
    --launcher none \
    --eval EPE \
    --test_seq_id $SEQ_ID

echo ""
echo "步骤2: 导出ONNX模型"
echo "----------------------------------------"
python tools/export_onnx.py \
    $CONFIG $CHECKPOINT \
    --output-dir work_dirs/onnx_models \
    --img-size 256 512 \
    --verify \
    --simplify

echo ""
echo "步骤3: ONNX推理 (CUDA加速)"
echo "----------------------------------------"
python tools/test_onnx.py \
    $CONFIG \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id $SEQ_ID \
    --eval EPE \
    --provider cuda

echo ""
echo "步骤4: ONNX推理 (TensorRT ExecutionProvider + FP16)"
echo "----------------------------------------"
python tools/test_onnx.py \
    $CONFIG \
    work_dirs/onnx_models/depth_net.onnx \
    --test_seq_id $SEQ_ID \
    --eval EPE \
    --provider tensorrt \
    --fp16

echo ""
echo "步骤5: TensorRT推理 (FP16精度)"
echo "----------------------------------------"
python tools/test_tensorrt.py \
    $CONFIG $CHECKPOINT \
    --test_seq_id $SEQ_ID \
    --eval EPE \
    --fp16 \
    --img-size 256 512

echo ""
echo "=========================================="
echo "完成! 请查看上述输出的性能统计"
echo "=========================================="
echo ""
echo "性能提升总结:"
echo "- ONNX (CUDA): ~1.5-2x 加速"
echo "- ONNX (TensorRT EP): ~2-3x 加速"
echo "- TensorRT (FP16): ~4-6x 加速"
echo ""
echo "推荐生产部署配置:"
echo "  TensorRT FP16 (最佳性能/精度平衡)"
echo ""
