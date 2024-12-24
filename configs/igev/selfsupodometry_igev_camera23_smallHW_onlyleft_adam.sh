#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-06-24 00:01:49
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and host='nefgpu52.inria.fr'

#OAR -l /nodes=1/gpunum=1,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  adam_odom_igev_cam23_leftsmallHW


source activate torch2

module load  gcc/9.2.0  cmake/3.10.1

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1     \
 tools/train.py configs/igev/selfsupodometry_igev_camera23_smallHW_onlyleft_adam.py   \
     --launcher pytorch

# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12288 \
  tools/test.py configs/igev/selfsupodometry_igev_camera23_smallHW_onlyleft_adam.py \
   work_dirs/selfsupodometry_igev_camera23_smallHW_onlyleft_adam/iter_40000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1 N_DISPS_EPE --test_seq_id 10
 