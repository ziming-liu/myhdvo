#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2023-03-23 09:25:38
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and gpucapability>='8.3'  

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

##OAR -t besteffort
 
#OAR --name  pyramid_base_setting_cascade11


source activate torch2


 
 
 #PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=16973 \
    tools/train.py configs/pyramid_stereo/pyramid_base_setting_cascade2.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}




#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=16973 \
    tools/train.py configs/pyramid_stereo/pyramid_base_setting_cascade3.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}



#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=16973 \
    tools/train.py configs/pyramid_stereo/pyramid_base_setting_cascade4.py \
      --launcher pytorch  --validate
# Any arguments from the third one are captured by ${@:3}

 

torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_base_setting_cascade2.py \
   work_dirs/pyramid_base_setting_cascade2/epoch_1.pth   \
    --launcher pytorch  --eval  EPE 3PE D1

torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_base_setting_cascade3.py \
   work_dirs/pyramid_base_setting_cascade3/epoch_1.pth   \
    --launcher pytorch  --eval  EPE 3PE D1

torchrun --standalone --nnodes=1 --nproc_per_node=3 --master_port=12200 \
  tools/test.py configs/pyramid_stereo/pyramid_base_setting_cascade4.py \
   work_dirs/pyramid_base_setting_cascade4/epoch_1.pth   \
    --launcher pytorch  --eval  EPE 3PE D1