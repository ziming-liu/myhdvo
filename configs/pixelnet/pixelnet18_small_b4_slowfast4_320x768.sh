#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-13 00:45:31
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and host='nefgpu52.inria.fr'  

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pixelnet18_small_b4_slowfast4_320x768


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2   \
    tools/train.py configs/pixelnet/pixelnet18_small_b4_slowfast4_320x768.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test.py configs/pixelnet/pixelnet18_small_b4_slowfast4_320x768.py \
   work_dirs/pixelnet18_small_b4_slowfast4_320x768/iter_48000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 

# 13006167 + resume 13006264 4k killed 28 k 

