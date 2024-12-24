"""
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_train.json
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_test.json

/home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_train_clean.json
"""

import os.path as osp
 
batch_size = 8
root2 = "/home/ziliu/mydata"

data_root = root2
annfile_root = osp.join(root2, "kitti_depth")


dataset_type = 'KITTIDepthEigenDataset'
 
img_norm_cfg = dict(mean=[88.78708011161852, 93.43778497818349, 91.33551888646076], std=[80.93941240862273, 81.55742718042109, 82.55097977909143], to_rgb=False)
crop_size=(224,704) # h, w (192,640) #
train_pipeline = [
    dict(type='LoadStereoImages', views=["left", "right"], to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True, to_gray=False), # let rgb to gray
    #dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoResize', scale=(1280,384), keep_ratio=False),
    #dict(type='StereoResize2', img_scale=(width,height), keep_ratio=True, ratio_range=(0.9, 1.1)),
    #dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    #dict(type='StereoFixedCrop', crop_size=(384-352, (1280-1216)//2 , 352, 1216)),
    #dict(type='VerticalCutDepth', ),
    #dict(type='HorizontalFlip', mono=True ),
    dict(type='StereoResize2', img_scale=(1024, 320), keep_ratio=False, ratio_range=(1,1)),
    #dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    dict(type='PhotoMetricDistortion'),
    #dict(type='StereoRandomCrop',size=(400,256)),
    #dict(type='Flip', flip_ratio=0),
    #dict(type='RandomBrightnessContrast', ),
    #dict(type='RandomGamma', ),
    #dict(type='HueSaturationValue', ),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'intrinsics','focal', 'baseline', 'pose'  ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',   'intrinsics','focal', 'baseline', 'pose'    ])
]
val_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True, to_gray=False),
    #dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(1280,384), keep_ratio=False),
    #dict(type='StereoPad', size=(416,1312), pad_val=0, disp_pad_val=0),
    #dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', ],crop_size=[256,1216]), #crop_size=[256,1216]), #crop_size=[256,1216]),
    #dict(type='ThreeCrop', crop_size=256),
    dict(type='StereoResize', scale=(800,256), keep_ratio=False),
    #dict(type='RGB2Gray', keys=['left_imgs', 'right_imgs', ]),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'intrinsics','focal', 'baseline',    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',   'intrinsics','focal', 'baseline',      ])
]
test_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True, to_gray=False),
    #dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(1280,384), keep_ratio=False),
    #dict(type='StereoPad', size=(416,1312), pad_val=0, disp_pad_val=0),
    #dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', ],crop_size=[256,1216]), #crop_size=[256,1216]), #crop_size=[256,1216]),
    #dict(type='ThreeCrop', crop_size=256),
    dict(type='StereoResize', scale=(1024,320), keep_ratio=False),
    #dict(type='RGB2Gray', keys=['left_imgs', 'right_imgs', ]),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',    ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'intrinsics','focal', 'baseline',  ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',   'intrinsics','focal', 'baseline',     ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_eigen_unsup_train_len3.json"),
        data_prefix=data_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        #end_id=10,
        test_mode= False,
        camera="23",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_eigen_unsup_test_seq.json"),
        end_id=10,
        data_prefix=data_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        test_mode= True,
        camera="23",
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_eigen_unsup_test_seq.json"),
        #end_id=100,
        data_prefix=data_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        crop_test_image='garg',
        test_mode= True,
        camera="23",
        pipeline=test_pipeline))