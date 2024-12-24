'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-09 17:09:34
LastEditors: Ziming Liu
LastEditTime: 2023-04-19 17:18:06
'''
import os.path as osp

batch_size = 8
root2 = "/home/ziliu/mydata"

data_root = osp.join(root2, 'sceneflow')
annfile_root = osp.join(root2, "sceneflow/annotations")
 
dataset_type = 'SceneFlowDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size=(256,384) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left",  ],  modalities=["disp"],),
    #dict(type='Augmentor', image_height=320,
    #                        image_width=768,
    #                        max_disp=192,
    #                        scale_min=0.6,
    #                        scale_max=1.0,
    #                        seed=0,),
    #dict(type='StereoColorJitter',
    #    img_keys=['left_imgs','right_imgs'],
    #    asymmetric_prob=0.2,
    #    brightness=0.4,
    #    contrast=0.4,
    #    saturation=0.4,
    #    hue=0.5 / 3.14),
    #dict(type='HorizontalFlip', p=0.5),
    dict(type='StereoResize', scale=(512,256), keep_ratio=False), # 960,544 original size
    dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    dict(type='PreLoadImageVolume', max_disp=192,mask_template=None),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=[ 'left_imgs', 'right_imgs', 'left_disps',        ]),
    dict(type='StereoFormatShape', input_format='NCDHW', keys=['image_volumes', ]),
    dict(type='Collect', keys=['image_volumes', 'left_disps', 'left_imgs', 'right_imgs',     ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['image_volumes', 'left_disps',   'left_imgs', 'right_imgs',    ])
]
val_pipeline = [
   dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", "right"],  modalities=["disp"],),
    #dict(type='StereoRandomCrop2', crop_size=(512,960), zeros_disp_max_ratio=1, random_shift=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoResize', scale=(512,256), keep_ratio=False), # 960,544 original size
    #dict(type='StereoPad', size=(576,960), pad_val=0, disp_pad_val=0),
    dict(type='PreLoadImageVolume', max_disp=192,mask_template=None),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=[ 'left_disps',  'left_imgs', 'right_imgs',      ]),
    dict(type='StereoFormatShape', input_format='NCDHW', keys=['image_volumes', ]),
    dict(type='Collect', keys=['image_volumes', 'left_disps',  'left_imgs', 'right_imgs',    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['image_volumes', 'left_disps',  'left_imgs', 'right_imgs',    ])
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
    dict(type='StereoResize', scale=(512,256), keep_ratio=False), # 960,544 original size
    #dict(type='StereoPad', size=(576,960), pad_val=0, disp_pad_val=0),
    dict(type='PreLoadImageVolume', max_disp=192,mask_template=None),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NCHW', keys=[ 'left_disps',  'left_imgs', 'right_imgs',      ]),
    dict(type='StereoFormatShape', input_format='NCDHW', keys=['image_volumes', ]),
    dict(type='Collect', keys=['image_volumes', 'left_disps',  'left_imgs', 'right_imgs',    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['image_volumes', 'left_disps',  'left_imgs', 'right_imgs',     ])
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
        end_id=1000,
        data_prefix=data_root,
        test_mode= True,
        pipeline=test_pipeline))