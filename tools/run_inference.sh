#!/bin/bash
# HDVO模型推理加速 - 快速启动脚本
# Author: ACENTAURI team
# Date: 2025-01-27

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
CONFIG="configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py"
CHECKPOINT="work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth"
SEQ_ID="09"
IMG_SIZE="256 512"
PROVIDER="cuda"

# 显示帮助信息
show_help() {
    cat << EOF
HDVO推理加速工具

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -c, --config PATH       配置文件路径 (默认: $CONFIG)
    -k, --checkpoint PATH   checkpoint文件路径 (默认: $CHECKPOINT)
    -s, --seq-id ID         测试序列ID (默认: $SEQ_ID)
    -m, --mode MODE         推理模式: pytorch|onnx|tensorrt (必需)
    -p, --provider PROV     ONNX provider: cuda|tensorrt|cpu (默认: $PROVIDER)
    --fp16                  使用FP16精度 (TensorRT)
    --benchmark             仅运行性能测试，不评估
    --export-only           仅导出模型，不进行推理

示例:
    # PyTorch推理
    $0 -m pytorch

    # ONNX推理 (CUDA)
    $0 -m onnx -p cuda

    # ONNX推理 (TensorRT EP + FP16)
    $0 -m onnx -p tensorrt --fp16

    # TensorRT推理 (FP16)
    $0 -m tensorrt --fp16

    # 仅导出ONNX模型
    $0 -m onnx --export-only

    # 性能对比
    $0 --benchmark
EOF
}

# 解析命令行参数
MODE=""
FP16=""
BENCHMARK=false
EXPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -k|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -s|--seq-id)
            SEQ_ID="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--provider)
            PROVIDER="$2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --export-only)
            EXPORT_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ "$BENCHMARK" = false ] && [ -z "$MODE" ]; then
    echo -e "${RED}错误: 必须指定推理模式 (-m/--mode)${NC}"
    show_help
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG${NC}"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}错误: Checkpoint文件不存在: $CHECKPOINT${NC}"
    exit 1
fi

# 运行Benchmark
if [ "$BENCHMARK" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}运行性能对比测试${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\n${YELLOW}[1/4] PyTorch推理...${NC}"
    OMP_NUM_THREADS=12 python tools/test.py \
        "$CONFIG" "$CHECKPOINT" \
        --launcher none \
        --test_seq_id "$SEQ_ID" \
        --eval EPE
    
    echo -e "\n${YELLOW}[2/4] ONNX推理 (CUDA)...${NC}"
    # 先导出ONNX
    if [ ! -f "work_dirs/onnx_models/depth_net.onnx" ]; then
        python tools/export_onnx.py "$CONFIG" "$CHECKPOINT" \
            --output-dir work_dirs/onnx_models \
            --img-size $IMG_SIZE
    fi
    python tools/test_onnx.py "$CONFIG" work_dirs/onnx_models/depth_net.onnx \
        --test_seq_id "$SEQ_ID" \
        --provider cuda \
        --eval EPE
    
    echo -e "\n${YELLOW}[3/4] ONNX推理 (TensorRT EP + FP16)...${NC}"
    python tools/test_onnx.py "$CONFIG" work_dirs/onnx_models/depth_net.onnx \
        --test_seq_id "$SEQ_ID" \
        --provider tensorrt \
        --fp16 \
        --eval EPE
    
    echo -e "\n${YELLOW}[4/4] TensorRT推理 (FP16)...${NC}"
    python tools/test_tensorrt.py "$CONFIG" "$CHECKPOINT" \
        --test_seq_id "$SEQ_ID" \
        --fp16 \
        --eval EPE
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}性能对比测试完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi

# PyTorch推理
if [ "$MODE" = "pytorch" ]; then
    echo -e "${GREEN}运行PyTorch推理...${NC}"
    OMP_NUM_THREADS=12 python tools/test.py \
        "$CONFIG" "$CHECKPOINT" \
        --launcher none \
        --eval EPE 3PE D1 \
        --test_seq_id "$SEQ_ID"
    
# ONNX推理
elif [ "$MODE" = "onnx" ]; then
    ONNX_MODEL="work_dirs/onnx_models/depth_net.onnx"
    
    # 导出ONNX模型
    if [ ! -f "$ONNX_MODEL" ]; then
        echo -e "${YELLOW}ONNX模型不存在，正在导出...${NC}"
        python tools/export_onnx.py \
            "$CONFIG" "$CHECKPOINT" \
            --output-dir work_dirs/onnx_models \
            --img-size $IMG_SIZE \
            --verify
    fi
    
    if [ "$EXPORT_ONLY" = true ]; then
        echo -e "${GREEN}ONNX模型已导出到: $ONNX_MODEL${NC}"
        exit 0
    fi
    
    echo -e "${GREEN}运行ONNX推理 (Provider: $PROVIDER)...${NC}"
    python tools/test_onnx.py \
        "$CONFIG" "$ONNX_MODEL" \
        --test_seq_id "$SEQ_ID" \
        --eval EPE 3PE D1 \
        --provider "$PROVIDER" \
        $FP16

# TensorRT推理
elif [ "$MODE" = "tensorrt" ]; then
    echo -e "${GREEN}运行TensorRT推理...${NC}"
    
    # 确定engine路径
    if [ -n "$FP16" ]; then
        ENGINE_PATH="work_dirs/depth_net_fp16.trt"
    else
        ENGINE_PATH="work_dirs/depth_net_fp32.trt"
    fi
    
    if [ "$EXPORT_ONLY" = true ]; then
        echo -e "${YELLOW}正在构建TensorRT engine...${NC}"
        # 导出会在第一次推理时自动进行
        echo -e "${GREEN}使用 test_tensorrt.py 进行首次推理时会自动构建engine${NC}"
        exit 0
    fi
    
    python tools/test_tensorrt.py \
        "$CONFIG" "$CHECKPOINT" \
        --test_seq_id "$SEQ_ID" \
        --eval EPE 3PE D1 \
        --engine-path "$ENGINE_PATH" \
        --img-size $IMG_SIZE \
        $FP16

else
    echo -e "${RED}错误: 不支持的推理模式: $MODE${NC}"
    echo -e "支持的模式: pytorch, onnx, tensorrt"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}推理完成!${NC}"
echo -e "${GREEN}========================================${NC}"
