#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-27 13:11:25
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpumem>=12000

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pyramid_base_setting_cascade8onekitti2012dirvingpretrain


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
OMP_NUM_THREADS=12 torchrun  --standalone --nnodes=1 --nproc_per_node=2  \
   tools/train.py configs/v8/pyramid_base_setting_cascade8onekitti2012dirvingpretrain.py \
      --launcher pytorch  --validate
 
#torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=12200 \
#  tools/test.py configs/v8/pyramid_base_setting_cascade8onekitti2012dirvingpretrain.py \
#   work_dirs/pyramid_base_setting_cascade8onekitti2012dirvingpretrain/iter_3000.pth   \
#    --launcher pytorch  --eval  EPE 3PE D1


 # 13001228 Done
 #  D1: 0.0795 3000iters

 # smaler lr 1e-5 13001426 Done

 # 13001616 R fix the bug of disp range