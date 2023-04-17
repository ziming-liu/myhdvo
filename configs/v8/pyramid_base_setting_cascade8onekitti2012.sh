#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-27 03:04:21
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpumem>=12000

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pyramid_base_setting_cascade8onekitti2012


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
OMP_NUM_THREADS=12 torchrun  --standalone --nnodes=1 --nproc_per_node=2  \
   tools/train.py configs/v8/pyramid_base_setting_cascade8onekitti2012.py \
      --launcher pytorch  --validate
 
##torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=12200 \
 # tools/test.py configs/v8/pyramid_base_setting_cascade8onekitti2012.py \
#   work_dirs/pyramid_base_setting_cascade8onekitti2012/iter_3000.pth   \
#    --launcher pytorch  --eval  EPE 3PE D1
 
 
 #13001227 Done

 # 13001427 200k iters 