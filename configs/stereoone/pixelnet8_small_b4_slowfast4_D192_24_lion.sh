#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2024-02-07 16:28:16
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and host='nefgpu54.inria.fr'

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pixelnet8_small_b4_slowfast4_D192_24_lion


source activate torch2


OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1  \
  tools/train.py configs/stereoone/pixelnet8_small_b4_slowfast4_D192_24_lion.py \
    --launcher pytorch  --validate

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test.py configs/stereoone/pixelnet8_small_b4_slowfast4_D192_24_lion.py \
   ckps/pixelnet8_small_b4_slowfast4_D192_24_lion.pth   \
    --launcher pytorch  --eval  EPE 3PE D1 N_DISPS_EPE --save_depth