<!--
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2024-02-02 14:57:55
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2024-02-02 16:45:08
-->
# Prepare your dataset


Note: the paths of image data and annotations are separated in config files. 

Option: Some dataset downloader script can be found at `tools/download_datasets`


## KITTI Odometry 

KITTI odometry dataset can be downloaded from [official site](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). You should 
- Download odometry data set (color, 65 GB)
- Download odometry data set (calibration files, 1 MB)
- Download odometry ground truth poses (4 MB)
at least. 

The dataset image data is organized as 

```
-KITTIodometry
   |--pose_GT
         |--00
         |--01
         |--...
         |--20
   |--sequences
         |--00
         |--01
         |--...
         |--20
```

The annotations are put under `HDVO/annotations/`. 

New annotations are generated with this script `tools/dataset_tools/kitti_odometry_annotation.py` 

```
import fire
python tools/dataset_tools/kitti_odometry_annotation.py save_path name  test_mode 
```

## KITTI Stereo


KITTI odometry dataset can be downloaded from [official site](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). You should 
- Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)
- Download calibration files (1 MB)
at least. 

```
-KITTI2015
     |--traning
          |--image_2
          |--image_3
          |--...
     |--testing
     |--calib
           |--training
           |--testing
                 |--calib_cam_to_cam
                 |--...
     
```

Stereo2012 version is similar. 


The annotations are put under `HDVO/annotations/`. 

New annotations are generated with this script 

```
python tools/dataset_tools/gen_kittistereo2015.py \
    --data-root /home/ziliu/mydata/kittistereo2015 \
        --save-annotation-root /home/ziliu/mydata/kittistereo2015/annotations \
            --is-full 

python tools/dataset_tools/gen_kittistereo2012.py \
    --data-root /home/ziliu/mydata/kittistereo2012/data_stereo_flow \
        --save-annotation-root /home/ziliu/mydata/kittistereo2012/annotations \
            --is-full 
```




## KITTI depth prediction

Download dataest from [site](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), you should at least 
- Downnload [raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- Download annotated depth maps data set (14 GB)
- Download manually selected validation and test data sets (2 GB)
- Download development kit (48 K)

This data is orgnized as 

```
--kitti_raw_data
           |--2011_09_26
           |--2011_09_28
           |--2011_09_29
           |--2011_09_30
           |--2011_10_03
                 |--2011_10_03_drive_0027_sync
                 |--...
```

The annotations are put under `HDVO/annotations/`. 

New annotations are generated with this script 

```
import fire
# 3 frame each sample
python tools/dataset_tools/kitti_eigen_annotation_len3.py  **kwargs

#  two frames each sample
tools/dataset_tools/kitti_eigen_annotation.py  **kwargs
```



## SceneFlow 

Download SceneFlow dataset from [site](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
-sceneflow
     |--driving
          |--frames_finalpass
          |--frames_cleanpass
          |--disparity
     |--flyingthings3d
     |--monkaa
```


The annotations are put under `HDVO/annotations/`. 

New annotations are generated with this script 

```
python  tools/dataset_tools/gen_sceneflow_anns.py --data-root   ../sceneflow  --save-annotation-root ../sceneflow/annotations --data-type  clean
```



