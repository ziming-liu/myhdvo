'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:09:34
LastEditors: Ziming Liu
LastEditTime: 2023-07-19 00:14:53

small batchsize for res101
'''
import os.path as osp

batch_size = 2
root2 = "/home/ziliu/mydata"

data_root = osp.join(root2, 'sceneflow')
annfile_root = osp.join(root2, "sceneflow/annotations")
 
dataset_type = 'SceneFlowDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size=(320,736) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left",],  modalities=["disp"],),
    dict(type='PhotoMetricDistortion'),
    dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',   ])
]
val_pipeline = [
   dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", "right"],  modalities=["disp"],),
    #dict(type='StereoRandomCrop2', crop_size=(512,960), zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(544,960), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ])
]
test_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", "right"],  modalities=["disp"],),
    #dict(type='StereoTopLeftCrop', crop_size=[512,960], keys=["left_imgs", "right_imgs", "left_disps"]),
    #dict(type='StereoRandomCrop2', crop_size=(512,960), zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(544,960), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_train_all.json"),
        data_prefix=data_root,
        eval_modality="disparity",
        eval_range=(1,192),
        #end_id=10,
        test_mode= False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_test.json"),
        data_prefix=data_root,
        eval_modality="disparity",
        eval_range=(1,192),
        end_id=100,
        test_mode= True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_test.json"),
        eval_modality="disparity",
        eval_range=(1,192),
        end_id=100,
        data_prefix=data_root,
        test_mode= True,
        pipeline=test_pipeline))