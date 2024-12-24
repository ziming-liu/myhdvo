'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-10 00:57:00
LastEditors: Ziming Liu
LastEditTime: 2023-09-14 01:59:38
'''
import os.path as osp

_base_ = [
    '../_base_/models/cascade_stereo.py',
    '../_base_/datasets/scene_flow.py', '../_base_/default_runtime.py',
    '../_base_/schedules/cascade_stereo_16epoch.py'
]

max_disp = 192
model=dict( 
        type="CascadeStereoGwcNet",
        maxdisp=max_disp,
        ndisps=[48,24],
        disp_interval_pixel=[4,1], 
        using_ns=True,  # using neighbor search
        ns_size=3, 
        grad_method="detach",
        cr_base_chs=[32, 32, 16], # cost regularization base channels
)

batch_size = 8
root2 = "/home/ziliu/mydata"

data_root = osp.join(root2, 'sceneflow')
annfile_root = osp.join(root2, "sceneflow/annotations")
 
dataset_type = 'SceneFlowDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size=(256,512) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left"],  modalities=["disp"],),
    dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]
val_pipeline = [
   dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left"],  modalities=["disp"],),
    dict(type='StereoRandomCrop2', crop_size=(512,960), zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',    ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]
test_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left"],  modalities=["disp"],),
    dict(type='StereoTopLeftCrop', crop_size=[512,960], keys=["left_imgs", "right_imgs", "left_disps"]),
    #dict(type='StereoRandomCrop2', crop_size=(512,960), zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
            dict(type='StereoResize', scale=(512,256), keep_ratio=False), # 960,544 original size

    #dict(type='StereoPad', size=(544,960), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',   ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'left_disps',  ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_train_all.json"),
        data_prefix=data_root,
        eval_range=(1,192),
        #end_id=100,
        test_mode= False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_test.json"),
        data_prefix=data_root,
        eval_range=(1,192),
        end_id=100,
        test_mode= True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_test.json"),
        eval_modality="disparity",
        eval_range=(1,192),
        #end_id=100,
        data_prefix=data_root,
        test_mode= True,
        pipeline=test_pipeline))

work_dir = "work_dirs/cacade_stereo_gwcnet"