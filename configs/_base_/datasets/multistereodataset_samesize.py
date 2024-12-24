
import os.path as osp
 
batch_size = 3
 
 
dataset_type = 'MultiStereoDataset'
 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size=(320,512) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='Resize', keys=['left_imgs', 'right_imgs', 'left_disps',], scale=(1280, 384), ratio_range=(0.8, 1.5), keep_ratio=True),
    #dict(type='Resize', scale=(650, 200)), # w, h 
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(1280, 384), keep_ratio=True),
    dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', 'left_disps',], crop_size=320),
    #dict(type='StereoResize2', img_scale=(width,height), keep_ratio=True, ratio_range=(0.9, 1.1)),
    dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    #dict(type='HorizontalFlip', mono=False),
    dict(type='PhotoMetricDistortion'),
    #dict(type='StereoRandomCrop',size=(400,256)),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',     ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps', ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]
val_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoResize', scale=(960,544), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(384,1280), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',     ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',    ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]
test_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', 'left_disps',], crop_size=320),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(384,1280), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(384,1280), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',     ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',    ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'left_disps',    ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=[ "/home/ziliu/mydata/drivingstereo/drivingstereo_half_train.json",\
                  "/home/ziliu/mydata/drivingstereo/drivingstereo_half_test.json",\
                   "/home/ziliu/mydata/kitti_depth/kitti_depth_sup_train.json",\
                   "/home/ziliu/mydata/kitti_depth/kitti_depth_sup_test.json",\
                    "/home/ziliu/mydata/kittistereo2015/annotations/split_train.json",\
                     "/home/ziliu/mydata/kittistereo2012/annotations/split_train.json"  ] ,
        data_prefix=[ "/home/ziliu/mydata/drivingstereo", "/home/ziliu/mydata/drivingstereo", "/home/ziliu/mydata", "/home/ziliu/mydata", \
                      "/home/ziliu/mydata/kittistereo2015", "/home/ziliu/mydata/kittistereo2012/data_stereo_flow"  ] ,
        eval_modality="disparity",
        depth_scale_ratio=[256, 256, 256, 256, 256, 256 ],
        #end_id=50,
        test_mode= False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=[ "/home/ziliu/mydata/kittistereo2015/annotations/split_eval.json",\
                     "/home/ziliu/mydata/kittistereo2012/annotations/split_eval.json"  ] ,
        #end_id=100,
        data_prefix=[
                      "/home/ziliu/mydata/kittistereo2015", "/home/ziliu/mydata/kittistereo2012/data_stereo_flow"  ] ,        
        eval_modality="disparity",
        depth_scale_ratio=[256, 256, ],
        test_mode= True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=[
                    "/home/ziliu/mydata/kittistereo2015/annotations/split_eval.json",\
                    # "/home/ziliu/mydata/kittistereo2012/annotations/split_eval.json"  
                     ] ,
        #end_id=100,
        data_prefix=[
                      "/home/ziliu/mydata/kittistereo2015", 
                      #"/home/ziliu/mydata/kittistereo2012/data_stereo_flow" 
                        ] ,                
        eval_modality="disparity",
        depth_scale_ratio=[256, 256, ],
        test_mode= True,
        pipeline=test_pipeline))