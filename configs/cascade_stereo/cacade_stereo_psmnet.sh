#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-25 16:26:14
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>='8.3'  

#OAR -l /nodes=1/gpunum=2,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  cacade_stereo_psmnet_sceneflow


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=16723 \
    tools/train.py configs/cascade_stereo/cacade_stereo_psmnet.py \
      --launcher pytorch  --validate --seed 1
# Any arguments from the third one are captured by ${@:3}

python -m torch.distributed.launch --nproc_per_node=2 --master_port=19200 \
  tools/test.py configs/cascade_stereo/cacade_stereo_psmnet.py \
   work_dirs/cacade_stereo_psmnet/epoch_1.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
  
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
 tools/test.py configs/cascade_stereo/cacade_stereo_psmnet.py \
   work_dirs/cacade_stereo_psmnet/epoch_16.pth   \
    --launcher pytorch  --eval  EPE 3PE D1