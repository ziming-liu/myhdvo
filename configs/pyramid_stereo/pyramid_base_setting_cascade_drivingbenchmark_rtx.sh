#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-23 11:47:51
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>='7.3'  

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  pyramid_base_setting_cascade_drivingbenchmark


module load cuda/11.0 cudnn/8.0-cuda-11.0 

source activate mmvo110

# gpucapability>8.3 
#NCCL_IB_DISABLE=1  NCCL_DEBUG=INFO OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=16973 \
#    tools/train.py configs/pyramid_stereo/pyramid_base_setting_cascade_drivingbenchmark.py \
#      --launcher pytorch  --validate

# gpucapability>7.3 <8.3 RTX 6000 RTX 8000
NCCL_IB_DISABLE=1   python -m torch.distributed.launch --nproc_per_node=3 \
 --master_port=16773     tools/train.py configs/pyramid_stereo/pyramid_base_setting_cascade_drivingbenchmark.py \
     --launcher pytorch  --validate

NCCL_IB_DISABLE=1  python -m torch.distributed.launch --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_base_setting_cascade_drivingbenchmark.py \
   work_dirs/pyramid_base_setting_cascade_drivingbenchmark/epoch_16.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
 