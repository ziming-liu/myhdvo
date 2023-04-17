#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-08 02:58:47
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>'8.1'

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pixelnet50_small_batch4_slowfast4_videopretrain


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2   \
    tools/train.py configs/pixelnet/pixelnet50_small_batch4_slowfast4_videopretrain.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=12200 \
  tools/test.py configs/pixelnet/pixelnet50_small_batch4_slowfast4_videopretrain.py \
   work_dirs/pixelnet50_small_batch4_slowfast4_videopretrain/iter_48000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 

 
