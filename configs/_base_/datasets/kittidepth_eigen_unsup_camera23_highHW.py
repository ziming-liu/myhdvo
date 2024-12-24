"""
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_train.json
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_test_seq.json

"""

import os.path as osp
 
batch_size = 3
root2 = "/home/ziliu/mydata"

data_root = root2
annfile_root = osp.join(root2, "kitti_depth")


dataset_type = 'KITTIDepthEigenDataset'
 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
crop_size=(256,768) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', views=["left", "right"], to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True, to_gray=False), # let rgb to gray
    #dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', ],crop_size=[320,1216]), #crop_size=[256,1216]),
    #dict(type='Resize', scale=(650, 200)), # w, h 
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    dict(type='StereoResize', scale=(1024,320), keep_ratio=False),
    #dict(type='StereoResize2', img_scale=(width,height), keep_ratio=True, ratio_range=(0.9, 1.1)),
   # dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    #dict(type='Augmentor', image_height=384,
    #                        image_width=512,
    #                        max_disp=256,
    #                        scale_min=0.6,
    #                        scale_max=1.0,
    #                        seed=0,),
    #dict(type='HorizontalFlip', mono=False),
    #dict(type='RGB2Gray', keys=['left_imgs', 'right_imgs', ]),
    #dict(type='PhotoMetricDistortion'),
    #dict(type='StereoRandomCrop',size=(400,256)),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs', 'intrinsics','focal', 'baseline', 'pose'  ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',   'raw_left_imgs','raw_right_imgs', 'intrinsics','focal', 'baseline', 'pose'    ])
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
    dict(type='StereoResize', scale=(1024,320), keep_ratio=False),
    #dict(type='RGB2Gray', keys=['left_imgs', 'right_imgs', ]),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs', 'intrinsics','focal', 'baseline',    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',  'intrinsics','focal', 'baseline',     ])
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
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs', 'intrinsics','focal', 'baseline', ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',  'intrinsics','focal', 'baseline', ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_eigen_unsup_train.json"),
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
        end_id=697,
        #end_id=11,
        data_prefix=data_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        test_mode= True,
        camera="23",
        pipeline=test_pipeline))