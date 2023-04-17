#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-17 11:28:18
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpumem>30000

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pyramid_stereo_light_weightedloss_smallrange_flying3d384x512_adamW2


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=16773 \
    tools/train.py configs/pyramid_stereo/pyramid_stereo_light_weightedloss_smallrange_flying3d384x512_adamW2.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_stereo_light_weightedloss_smallrange_flying3d384x512_adamW2.py \
   work_dirs/pyramid_stereo_light_weightedloss_smallrange_flying3d384x512_adamW2/iter_300000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
 