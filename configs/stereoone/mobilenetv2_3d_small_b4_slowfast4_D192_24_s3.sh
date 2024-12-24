#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2024-02-07 21:16:25
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and  host='nefgpu52.inria.fr'

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  mobilenetv2_3d_small_b4_slowfast4_D192_24_s3


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2  \
    tools/train.py configs/stereoone/mobilenetv2_3d_small_b4_slowfast4_D192_24_s3.py \
     --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/stereoone/mobilenetv2_3d_small_b4_slowfast4_D192_24_s3.py \
  ckps/mobilenetv2_3d_small_b4_slowfast4_d192_24_s3.pth  \
    --launcher pytorch  --eval  EPE 3PE D1 N_DISPS_EPE  

#OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
#  tools/test_flops_counter.py configs/stereoone/mobilenetv2_3d_small_b4_slowfast4_D192_24_s3.py \
#   work_dirs/mobilenetv2_3d_small_b4_slowfast4_D192_24_s3/iter_144000.pth   \
#    --launcher pytorch --input_size 544,960

#python tools/deploy.py configs/stereoone/mobilenetv2_3d_small_b4_slowfast4_D192_24_s3.py \
#   work_dirs/mobilenetv2_3d_small_b4_slowfast4_d192_24_s3/iter_200000.pth   \
#    --launcher pytorch  --eval  EPE 3PE D1 N_DISPS_EPE  
 