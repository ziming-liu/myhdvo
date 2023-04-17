#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-11 00:55:12
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>'8.2'

#OAR -l /nodes=1/gpunum=1,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pixelnet101_small_batch4_slowfast4_videopretrain


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1   \
    tools/train.py configs/pixelnet/pixelnet101_small_batch4_slowfast4_videopretrain.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test.py configs/pixelnet/pixelnet101_small_batch4_slowfast4_videopretrain.py \
   work_dirs/pixelnet101_small_batch4_slowfast4_videopretrain/iter_48000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 

 
