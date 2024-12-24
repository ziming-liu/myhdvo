<!--
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2024-02-07 12:30:08
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2024-02-08 17:34:23
-->
# HDVO

Visual odometry is an important part of the perception module of autonomous robots. Recent advances in deep learning approaches have given rise to hybrid visual odometry approaches that combine both deep networks and traditional pose estimation methods. One limitation of deep learning approaches is the availability of ground truth data needed to train the neural networks. For example, it is extremely difficult, if not impossible, to obtain a ground truth dense depth map of the environment to be used for stereo visual odometry. Even if unsupervised training of networks has been investigated, supervised training remains more reliable and robust. In this paper, we propose a new hybrid dense stereo visual odometry approach in which a dense depth map is obtained with a network that is supervised using ground truth poses that can be more easily obtained than ground truth depths maps. The depth map obtained from the neural network is used to warp the current image into the reference frame and the optimal pose is obtained by minimizing a cost function that encodes the similarity between the warped image and the reference image. The experimental results show that the proposed approach, not only improves state-of-the-art depth maps estimation networks on some of the standard benchmark datasets, but also outperforms the state-of-the-art visual odometry methods. 


Three models are provided. 

- psmnet backbone + L1 loss 
- psmnet backbone + huber loss
- coexnet backbone + huber loss 

｜model|backbone|loss| error t(%)| error R(deg/100m)| ATE | RPE(m)|RPE(deg)|weights|
|------|--------|----|----------|-------------------|------|------|-------|--------|
|HDVO| coex| huber | 0.98|0.36|0.0098std0.0089|0.013|0.039| [weights](ckps/stereohdvo_posesup_coex_kittiodom_huberloss.pth) | 
|HDVO|psmnet|huber|0.65|0.26|0.0097std0.0089|0.012|0.039 | [weights](ckps/stereohdvo_posesup_s1_kittiodom_huberloss.pth)|
|HDVO|psmnet|l1 |1.11|0.51|0.0109std0.0090|0.014|0.044  | [weights](ckps/stereohdvo_posesup_s1_kittiodom.pth)|




## training 

```

 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1  \
    tools/train.py configs/hdvo/stereohdvo_posesup_s1_kittiodom_huberloss.py \
      --launcher pytorch --validate
```

## testing 

```

OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12860 \
  tools/test.py configs/hdvo/stereohdvo_posesup_s1_kittiodom_huberloss.py \
   work_dirs/stereohdvo_posesup_s1_kittiodom_huberloss/iter_40000.pth   \
    --launcher pytorch  --eval  EPE 3PE D1  --save_depth

```


## citation

```

@inproceedings{HDVO-IROS-2022,
  title={A New Dense Hybrid Stereo Visual Odometry Approach},
  author={Liu, Ziming and Malis, Ezio and Martinet, Philippe},
  booktitle={IROS},
  pages={6998--7003},
  year={2022},
  organization={IEEE}
}



@inproceedings{chang2018pyramid,
  title={Pyramid stereo matching network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5410--5418},
  year={2018}
}


@inproceedings{bangunharcana2021correlate,
  title={Correlate-and-excite: Real-time stereo matching via guided cost volume excitation},
  author={Bangunharcana, Antyanta and Cho, Jae Won and Lee, Seokju and Kweon, In So and Kim, Kyung-Soo and Kim, Soohyun},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3542--3548},
  year={2021},
  organization={IEEE}
}



```
