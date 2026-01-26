#!/bin/bash
# HDVO FP16推理测试示例
# 展示如何在原有的测试pipeline中启用FP16加速

# ==============================================================================
# 配置参数
# ==============================================================================
CONFIG="configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py"
CHECKPOINT="work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth"

# ==============================================================================
# 示例1: 基本FP16推理（与原始命令对比）
# ==============================================================================
echo "=========================================="
echo "示例1: 基本FP16推理"
echo "=========================================="

# 原始FP32命令
echo "原始FP32推理:"
echo "OMP_NUM_THREADS=12 python tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher none \\"
echo "    --eval EPE 3PE D1 \\"
echo "    --test_seq_id 09 \\"
echo "    --no_gt"
echo ""

# FP16加速版本 - 只需添加 --fp16
echo "FP16加速推理 (添加 --fp16):"
echo "OMP_NUM_THREADS=12 python tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher none \\"
echo "    --eval EPE 3PE D1 \\"
echo "    --test_seq_id 09 \\"
echo "    --no_gt \\"
echo "    --fp16  # <-- 只需添加此参数！"
echo ""

# ==============================================================================
# 示例2: 保存深度图的FP16推理
# ==============================================================================
echo "=========================================="
echo "示例2: 保存深度图的FP16推理"
echo "=========================================="

echo "OMP_NUM_THREADS=12 python tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher none \\"
echo "    --eval EPE 3PE D1 \\"
echo "    --test_seq_id 09 \\"
echo "    --save_depth \\"
echo "    --fp16"
echo ""

# ==============================================================================
# 示例3: 分布式多GPU FP16推理
# ==============================================================================
echo "=========================================="
echo "示例3: 分布式多GPU FP16推理"
echo "=========================================="

echo "OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=12860 \\"
echo "    tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher pytorch \\"
echo "    --eval EPE 3PE D1 \\"
echo "    --test_seq_id 09 \\"
echo "    --fp16"
echo ""

# ==============================================================================
# 示例4: 测试特定帧范围
# ==============================================================================
echo "=========================================="
echo "示例4: 测试特定帧范围（快速验证）"
echo "=========================================="

echo "OMP_NUM_THREADS=12 python tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher none \\"
echo "    --eval EPE 3PE D1 \\"
echo "    --test_seq_id 09 \\"
echo "    --test_range 0:100 \\"
echo "    --no_gt \\"
echo "    --fp16"
echo ""

# ==============================================================================
# 示例5: 保存和加载预测结果
# ==============================================================================
echo "=========================================="
echo "示例5: 保存预测结果为PKL文件"
echo "=========================================="

echo "# 第一步: 运行FP16推理并保存结果"
echo "OMP_NUM_THREADS=12 python tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher none \\"
echo "    --test_seq_id 09 \\"
echo "    --save_pkl \\"
echo "    --fp16"
echo ""
echo "# 第二步: 从保存的PKL文件评估（无需重新推理）"
echo "python tools/test.py \\"
echo "    $CONFIG \\"
echo "    $CHECKPOINT \\"
echo "    --launcher none \\"
echo "    --eval EPE 3PE D1 \\"
echo "    --test_seq_id 09 \\"
echo "    --output_pkl work_dirs/.../outputs.pkl"
echo ""

# ==============================================================================
# 示例6: 在BSUB脚本中使用（集群任务）
# ==============================================================================
echo "=========================================="
echo "示例6: 在BSUB任务脚本中集成FP16"
echo "=========================================="

cat << 'EOF'
#!/bin/bash -l
# BSUB -J hdvo_fp16_test
# BSUB -o logs/hdvo_fp16.%J.stdout
# BSUB -e logs/hdvo_fp16.%J.stderr
# BSUB -q gpu_a100_batch
# BSUB -W 2:00
# BSUB -n 4
# BSUB -M 20480
# BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/11.8.0
conda activate hdvo

# 使用FP16加速推理
OMP_NUM_THREADS=12 python tools/test.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id 09 \
    --fp16
EOF

echo ""

# ==============================================================================
# 性能对比总结
# ==============================================================================
echo "=========================================="
echo "性能对比总结"
echo "=========================================="
cat << 'EOF'

基于Nvidia Jetson Orin的测试结果:

┌──────────────┬──────────┬────────┬──────────┐
│ 推理方式     │ 延迟     │ FPS    │ 加速比   │
├──────────────┼──────────┼────────┼──────────┤
│ FP32原始     │ 54.26ms  │ 18.43  │ 1.00x    │
│ FP16加速     │ 42.98ms  │ 23.27  │ 1.26x ⭐ │
│ JIT编译      │ 44.98ms  │ 22.23  │ 1.21x    │
└──────────────┴──────────┴────────┴──────────┘

推荐: 使用FP16获得最佳推理速度！

EOF

echo ""
echo "=========================================="
echo "更多信息"
echo "=========================================="
echo "详细文档: docs/FP16_INFERENCE_GUIDE.md"
echo "性能报告: work_dirs/INFERENCE_ACCELERATION_REPORT.md"
echo "测试脚本: tools/test_fp16.sh"
echo "=========================================="
