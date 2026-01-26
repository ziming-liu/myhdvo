#!/bin/bash
# Quick test script for FP16 inference with HDVO

echo "=========================================="
echo "Testing HDVO with FP16 Acceleration"
echo "=========================================="

# Test parameters
CONFIG="configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py"
CHECKPOINT="work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth"
TEST_SEQ="09"

echo ""
echo "1. Running baseline PyTorch (FP32) inference..."
echo "----------------------------------------------"
OMP_NUM_THREADS=12 python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id $TEST_SEQ \
    --no_gt

echo ""
echo ""
echo "2. Running FP16 accelerated inference..."
echo "----------------------------------------------"
OMP_NUM_THREADS=12 python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher none \
    --eval EPE 3PE D1 \
    --test_seq_id $TEST_SEQ \
    --no_gt \
    --fp16

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
