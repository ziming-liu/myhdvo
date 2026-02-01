

import os.path as osp
 
batch_size = 1

raw_kitti_root = "data_sets/Kitti-Dataset"

data_root = "data_sets/kitti_odometry"
annfile_root = osp.join("annotations", "kittiodometry")
 

dataset_type = 'KITTIOdometryDataset'
 
img_norm_cfg = dict(mean=[88.78708011161852, 93.43778497818349, 91.33551888646076], std=[80.93941240862273, 81.55742718042109, 82.55097977909143], to_rgb=False)
crop_size=(320,768) # h, w
RESIZE_SIZE = (1024,320)
train_pipeline = [
    dict(type='LoadStereoImages', views=["left", "right"], to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True, to_gray=False), # let rgb to gray
    #dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', ],crop_size=[320,1216]), #crop_size=[256,1216]),
    #dict(type='Resize', scale=(650, 200)), # w, h 
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=RESIZE_SIZE, keep_ratio=False),
    dict(type='StereoResize2', img_scale=RESIZE_SIZE, keep_ratio=False, ratio_range=(1, 1.)),
    #dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
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
    dict(type='Collect', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs', 'intrinsics', 'focal', 'baseline', 'pose'  ], meta_keys=['left_frame_paths'  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',   'raw_left_imgs','raw_right_imgs', 'intrinsics', 'focal', 'baseline', 'pose'    ])
]
val_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True, to_gray=False),
    #dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoResize', scale=(960,544), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    #dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', ],crop_size=[256,1216]), #crop_size=[256,1216]), #crop_size=[256,1216]),
    dict(type='StereoResize', scale=RESIZE_SIZE, keep_ratio=False),
    #dict(type='StereoResize', scale=(1280,384), keep_ratio=False),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',   'raw_left_imgs','raw_right_imgs',  ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs', 'intrinsics', 'focal', 'baseline', 'pose'  ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',   'raw_left_imgs','raw_right_imgs', 'intrinsics', 'focal', 'baseline', 'pose'    ])
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
   # dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', ],crop_size=[256,1216]), #crop_size=[256,1216]), #crop_size=[256,1216]),
    dict(type='StereoResize2', img_scale=RESIZE_SIZE, keep_ratio=False, ratio_range=(1, 1.)),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='RGB2Gray', keys=['left_imgs', 'right_imgs', ]),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=(416,1312), pad_val=0, disp_pad_val=0),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs', 'intrinsics', 'focal', 'baseline', 'pose'  ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'raw_left_imgs','raw_right_imgs',  'intrinsics', 'focal', 'baseline', 'pose'    ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_odometry_train_3frame.json"),
        data_prefix=data_root,
        kitti_rawdata_path=raw_kitti_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        #end_id=10,
        test_mode= False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_odometry_test_09.json"),
        end_id=10,
        data_prefix=data_root,
        kitti_rawdata_path=raw_kitti_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        test_mode= True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "kitti_odometry_test_09.json"),
        #end_id=100,
        test_seq_id=9,
        data_prefix=data_root,
        kitti_rawdata_path=raw_kitti_root,
        eval_modality="depth",
        depth_scale_ratio=256,
        test_mode= True,
        pipeline=test_pipeline))