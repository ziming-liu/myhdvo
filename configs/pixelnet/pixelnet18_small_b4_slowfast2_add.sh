#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-06 21:28:11
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpumem>30000

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pixelnet18_small_b4_slowfast2_add


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2   \
    tools/train.py configs/pixelnet/pixelnet18_small_b4_slowfast2_add.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=12200 \
  tools/test.py configs/pixelnet/pixelnet18_small_b4_slowfast2_add.py \
   work_dirs/pixelnet18_small_b4_slowfast2_add/iter_48000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
# 13006267 + 13006281 resume
