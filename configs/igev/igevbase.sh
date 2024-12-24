#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-23 18:29:31
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and  host='nefgpu53.inria.fr'  

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

##OAR -t besteffort
 
#OAR --name  igevbase


source activate torch2


 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=16773 \
    tools/train.py configs/igev/igevbase.py \
      --launcher pytorch  --seed 666
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/igev/igevbase.py \
   work_dirs/igevbase/iter_150000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
 