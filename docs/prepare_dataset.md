
# Prepare your dataset


Note: the paths of image data and annotations are separated in config files. 

Option: Some dataset downloader script can be found at `tools/download_datasets`


## KITTI Odometry 

KITTI odometry dataset can be downloaded from [official site](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). You should 
- Download odometry data set (color, 65 GB)
- Download odometry data set (calibration files, 1 MB)
- Download odometry ground truth poses (4 MB)
at least. 

The dataset image data is organized as:

```
KITTIodometry/
├── pose_GT/
│   ├── 00/
│   ├── 01/
│   ├── ...
│   └── 20/
└── sequences/
    ├── 00/
    ├── 01/
    ├── ...
    └── 20/
```

Then link kitti odometry dataset to `data_sets/kitti_doometry`.

The annotations are put under `annotations/kittiodometry/`. 

New annotations are generated with this script:

```bash
python tools/dataset_tools/kitti_odometry_annotation.py <save_path> <name> <test_mode>
```

## VKitti2

VKitti2 (Virtual KITTI 2) dataset can be downloaded from [official site](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/). This is a synthetic dataset with ground truth depth and camera poses.

The dataset is organized as:

```
vkitti2/
├── Scene01/
│   ├── clone/
│   ├── 15-deg-left/
│   ├── 15-deg-right/
│   ├── 30-deg-left/
│   ├── 30-deg-right/
│   ├── fog/
│   ├── morning/
│   ├── overcast/
│   ├── rain/
│   └── sunset/
│       ├── frames/
│       │   ├── rgb/
│       │   │   ├── Camera_0/
│       │   │   └── Camera_1/
│       │   └── depth/
│       │       ├── Camera_0/
│       │       └── Camera_1/
│       ├── intrinsic.txt
│       ├── extrinsic.txt
│       ├── pose.txt
│       ├── bbox.txt
│       ├── colors.txt
│       └── info.txt
├── Scene02/
├── Scene06/
├── Scene18/
└── Scene20/
```


The annotations are put under `annotations/vkitti2/`.

New annotations are generated with this script `tools/dataset_tools/vkitti2_annotation.py`:

```bash
# Generate annotations for specific scene and variation
python tools/dataset_tools/vkitti2_annotation.py

# The script can be modified to generate annotations for different scenes and variations
# Available scenes: Scene01, Scene02, Scene06, Scene18, Scene20
# Available variations: clone, 15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right,
#                       fog, morning, overcast, rain, sunset
```

Test the dataloader:
```bash
python test_vkitti2_loader.py
```

## KITTI Stereo

KITTI Stereo dataset can be downloaded from [official site](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php). You should 
- Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)
- Download calibration files (1 MB)
at least. 

```
KITTI2015/
├── training/
│   ├── image_2/
│   ├── image_3/
│   └── ...
├── testing/
└── calib/
    ├── training/
    └── testing/
        ├── calib_cam_to_cam/
        └── ...
```

Note: KITTI Stereo 2012 has a similar structure. 


The annotations are put under `annotations/kittistereo2015/` and `annotations/kittistereo2012/`.

New annotations are generated with these scripts:

```bash
# KITTI Stereo 2015
python tools/dataset_tools/gen_kittistereo2015.py \
    --data-root <path_to_kittistereo2015> \
    --save-annotation-root <path_to_annotations> \
    --is-full

# KITTI Stereo 2012
python tools/dataset_tools/gen_kittistereo2012.py \
    --data-root <path_to_kittistereo2012>/data_stereo_flow \
    --save-annotation-root <path_to_annotations> \
    --is-full
```




## KITTI depth prediction

Download dataest from [site](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), you should at least 
- Downnload [raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- Download annotated depth maps data set (14 GB)
- Download manually selected validation and test data sets (2 GB)
- Download development kit (48 K)

The data is organized as:

```
kitti_raw_data/
├── 2011_09_26/
├── 2011_09_28/
├── 2011_09_29/
├── 2011_09_30/
└── 2011_10_03/
    ├── 2011_10_03_drive_0027_sync/
    └── ...
```

Then link kitti raw data to `data_sets/Kitti-Dataset`.

The annotations are put under `annotations/`.

New annotations are generated with these scripts:

```bash
# 3 frames per sample
python tools/dataset_tools/kitti_eigen_annotation_len3.py <arguments>

# 2 frames per sample
python tools/dataset_tools/kitti_eigen_annotation.py <arguments>
```



## SceneFlow 

Download SceneFlow dataset from [site](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
sceneflow/
├── driving/
│   ├── frames_finalpass/
│   ├── frames_cleanpass/
│   └── disparity/
├── flyingthings3d/
└── monkaa/
```

The annotations are put under `annotations/sceneflow/`.

New annotations are generated with this script:

```bash
python tools/dataset_tools/gen_sceneflow_anns.py \
    --data-root <path_to_sceneflow> \
    --save-annotation-root <path_to_annotations> \
    --data-type clean
```



