#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-19 10:39:38
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpumem>10000

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  sceneflow_psmnet_baseline


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1   \
    tools/train.py configs/psmnet/psmnet_small_b4.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=12200 \
  tools/test.py configs/psmnet/psmnet_small_b4.py \
   work_dirs/psmnet_small_b4/iter_25000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 