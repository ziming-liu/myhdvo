#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-27 03:10:45
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and host='nefgpu52.inria.fr'

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pyramid_base_setting_cascade8oneSF200k10iters2


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    tools/train.py configs/pyramid_stereo/pyramid_base_setting_cascade8oneSF200k10iters2.py \
      --launcher pytorch  --validate
 
torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_base_setting_cascade8oneSF200k10iters2.py \
   work_dirs/pyramid_base_setting_cascade8oneSF200k10iters2/iter_200000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
 
# 13001293 R

