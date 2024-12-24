#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2024-02-08 17:38:16
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpumem>30000

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pixelnet18_small_b8


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2   \
    tools/train.py configs/stereoone/pixelnet18_small_b8.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/stereoone/pixelnet18_small_b8.py \
   ckps/pixelnet18_small_b8.pth   \
    --launcher pytorch  --eval  EPE 3PE D1

 