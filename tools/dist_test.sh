#!/usr/bin/env bash
###
 # @Developer: ACENTAURI team, INRIA institute
 # @Author: Ziming Liu
 # @Date: 2021-03-02 20:41:04
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-18 18:11:20
### 

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS  --master_port=$PORT  \
$(dirname "$0")/test.py  $CONFIG  $CHECKPOINT --launcher pytorch ${@:4} 

