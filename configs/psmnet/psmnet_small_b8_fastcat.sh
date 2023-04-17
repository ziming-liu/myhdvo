#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-11 00:25:19
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>'8.2'

#OAR -l /nodes=1/gpunum=1,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  psmnet_small_b8_fastcat


source activate torch2


OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1   \
    tools/train.py configs/psmnet/psmnet_small_b8_fastcat.py \
      --launcher pytorch  --validate


OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test.py configs/psmnet/psmnet_small_b8_fastcat.py \
   work_dirs/psmnet_small_b8_fastcat/iter_25000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 

# 13004630

# test 13006546