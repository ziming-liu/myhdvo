#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2024-02-08 17:21:57
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and  host='nefgpu55.inria.fr'

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  stereohdvo_posesup_s1_kittiodom


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1  \
    tools/train.py configs/hdvo/stereohdvo_posesup_s1_kittiodom.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}


OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12660 \
  tools/test.py configs/hdvo/stereohdvo_posesup_s1_kittiodom.py \
   ckps/stereohdvo_posesup_s1_kittiodom.pth   \
    --launcher pytorch  --eval  EPE 3PE D1 # --save_depth


#python tools/deploy.py configs/hdvo/stereohdvo_posesup_s1_kittiodom.py \
#   work_dirs/stereohdvo_posesup_s1_kittiodom/iter_40000.pth   \
#    --launcher none  --eval  EPE 3PE D1  --save_depth
 