#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-07-19 11:39:03
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and host='nefgpu56.inria.fr'

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  coex


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2   \
    tools/train.py configs/coex/coex.py \
     --launcher pytorch 
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test.py configs/coex/coex.py \
   work_dirs/coex/iter_48000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1 N_DISPS_EPE #--save_depth

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test_flops_counter.py configs/coex/coex.py \
   work_dirs/coex/iter_10.pth   \
    --launcher pytorch --input_size 544,960
 
 