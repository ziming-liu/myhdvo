#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-22 14:03:36
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>='8.3'  

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

##OAR -t besteffort
 
#OAR --name  pyramid_stereov1_SF


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=16973 \
    tools/train.py configs/pyramid_stereo/pyramid_stereov1_SF.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_stereov1_SF.py \
   work_dirs/pyramid_stereov1_SF/iter_16.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
 