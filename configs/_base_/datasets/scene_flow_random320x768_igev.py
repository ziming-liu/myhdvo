'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:09:34
LastEditors: Ziming Liu
LastEditTime: 2023-04-27 10:56:25
'''
import os.path as osp

batch_size = 4
root2 = "/home/ziliu/mydata"

data_root = osp.join(root2, 'sceneflow')
annfile_root = osp.join(root2, "sceneflow/annotations")
 
dataset_type = 'SceneFlowDataset'
img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size=(320,736) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True),
    dict(type='LoadAnnotations', views=["left",  ],  modalities=["disp"],),
    #dict(type='Augmentor', image_height=320,
    #                        image_width=768,
    #                        max_disp=192,
    #                        scale_min=0.6,
    #                        scale_max=1.0,
    #                        seed=0,),
    dict(type='FlowAugmentor', crop_size=crop_size, min_scale=-0.2, max_scale=0.4, do_flip=False, 
         yjitter=True, saturation_range=[0,1.4], gamma=[1,1,1,1]),
    #dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    #dict(type='PhotoMetricDistortion'),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',      ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',     ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',      ])
]
val_pipeline = [
   dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',to_rgb=True),
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
                 imdecode_backend='cv2',to_rgb=True),
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
        init_seed=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_test.json"),
        data_prefix=data_root,
        eval_modality="disparity",
        eval_range=(1,192),
        end_id=100,
        test_mode= True,
        init_seed=False,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "finalpass_test.json"),
        eval_modality="disparity",
        eval_range=(0.5,192),
        #end_id=100,
        data_prefix=data_root,
        test_mode= True,
        init_seed=False,
        pipeline=test_pipeline))