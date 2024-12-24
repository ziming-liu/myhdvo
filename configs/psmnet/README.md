<!--
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2024-02-02 22:58:54
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2024-02-08 17:40:42
-->

## abstract
Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural net- works (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to ex- ploit context information for finding correspondence in ill- posed regions. To tackle this problem, we propose PSM- Net, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the ca- pacity of global context information by aggregating con- text in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with interme- diate supervision. The proposed approach was evaluated on several benchmark datasets. Our method ranked first in the KITTI 2012 and 2015 leaderboards before March 18, 2018. 




## models 

- supervised results 
- self-supervised results


|model|loss|weights|
|-----|-----|------|
|psmnet|sup-L1| [weights](ckps/psmnet_sup.pth)|
|psmnet|self-sup| [weights](ckps/psmnet_selfsup.pth)|




## training & test 

- psmnet supervised model

```
# Training 

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1   \
    tools/train.py configs/psmnet/psmnet_sup.py \
      --launcher pytorch   --validate

# Test 

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12200 \
  tools/test.py configs/psmnet/psmnet_sup.py\
   ckps/psmnet_sup.pth  \
    --launcher pytorch  --eval  EPE 3PE D1
 

```




- psmnet self-supervised model 


```

# training 

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1     \
 tools/train.py configs/psmnet/psmnet_selfsup.py   \
     --launcher pytorch

# test 

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12688 \
  tools/test.py configs/psmnet/psmnet_selfsup.py \ ckps/psmnet_selfsup.pth   \
    --launcher pytorch  --eval  EPE 3PE D1 N_DISPS_EPE #--save_depth
 

```