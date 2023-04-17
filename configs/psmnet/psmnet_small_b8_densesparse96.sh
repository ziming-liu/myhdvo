#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-04-02 11:41:02
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and (host='nefgpu52.inria.fr'  or host='nefgpu55.inria.fr' or host='nefgpu54.inria.fr' or host='nefgpu53.inria.fr') 

#OAR -l /nodes=1/gpunum=1,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  sceneflow_psmnet_baseline


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1   \
    tools/train.py configs/psmnet/psmnet_small_b8_densesparse96.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=12200 \
  tools/test.py configs/psmnet/psmnet_small_b8_densesparse96.py \
   work_dirs/psmnet_small_b8_densesparse96/iter_47000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1
 
 # OAR_JOB_ID=13004475