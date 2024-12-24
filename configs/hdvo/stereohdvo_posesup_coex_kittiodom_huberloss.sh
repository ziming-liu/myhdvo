#!/usr/bin/env bash
###
 # @Author: Ziming Liu
 # @Date: 2022-05-09 01:06:59
 # @LastEditors: Ziming Liu
 # @LastEditTime: 2024-12-24 23:51:01
 # @Description: ...
 # @Dependent packages: don't need any extral dependency
### 

#OAR -p gpu='YES' and  host='nefgpu55.inria.fr'

#OAR -l /nodes=1/gpunum=3,walltime=48:00:00

#OAR -t besteffort
 
#OAR --name  stereohdvo_posesup_coex_kittiodom_huberloss


source activate torch2


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1  \
    tools/train.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
      --launcher pytorch --validate
# Any arguments from the third one are captured by ${@:3}


OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12860 \
  tools/test.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
   ckps/stereohdvo_posesup_coex_kittiodom_huberloss.pth   \
    --launcher pytorch  --eval  EPE 3PE D1  --test_seq_id 09 # --save_depth
 
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12860 \
  tools/test_time_training.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
   ckps/stereohdvo_posesup_coex_kittiodom_huberloss.pth   \
    --launcher pytorch  --eval  EPE 3PE D1  --test_seq_id 09  --seq_optim 20 --lr 5e-6  # --save_depth
 


#python tools/deploy.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
#   work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth   \
#    --launcher pytorch  --eval  EPE 3PE D1 # --save_depth
 



 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12860 \
  tools/test.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    /home/ziliu/zimingdepth_torch2/work_dirs/coex/iter_48000.pth \
    --launcher pytorch  --eval  EPE 3PE D1  --test_seq_id 09 # --save_depth
 
 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12060 \
  tools/test_time_training.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
   /home/ziliu/zimingdepth_torch2/work_dirs/coex/iter_48000.pth \
    --launcher pytorch  --eval  EPE 3PE D1  --test_seq_id 09  --seq_optim 20 --lr 5e-6  # --save_depth
 