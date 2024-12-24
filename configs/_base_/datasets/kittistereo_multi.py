
import os.path as osp
 
batch_size = 4
data_root = "/home/ziliu/mydata"

data_prefix = [data_root,
               osp.join(data_root, "kittistereo2012/data_stereo_flow"),
               osp.join(data_root, "kittistereo2015"),]

train_annfiles = [osp.join(data_root, "kitti_depth", "kitti_depth_sup_train.json"),
                   osp.join(data_root, "kittistereo2012/annotations/split_train.json"),
                   osp.join(data_root, "kittistereo2015/annotations/split_train.json"),]

test_annfiles = [osp.join(data_root, "kitti_depth", "kitti_depth_sup_test.json"),
                   osp.join(data_root, "kittistereo2012/annotations/split_eval.json"),
                   osp.join(data_root, "kittistereo2015/annotations/split_eval.json"),]


dataset_type = 'KITTIStereoDataset'
#data_root = '/data/acentauri/user/ziliu/data/'
#data_root_val = 'data/kinetics400/rawframes_val'
#ann_file_train = '/home/ziliu/DeepVO-pytorch/KITTI/base_train_more.pickle'
#ann_file_val = '/home/ziliu/DeepVO-pytorch/KITTI/base_val.pickle'
#ann_file_test = '/home/ziliu/DeepVO-pytorch/KITTI/base_test.pickle'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size=(256,512) # h, w
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    dict(type='KITTIKBCrop', keys=['left_imgs', 'right_imgs', 'left_disps',], crop_size=[256,1216]),
    #dict(type='RandomResize', keys=['left_imgs', 'right_imgs', 'left_disps',], scale=(356, 1024), ratio_range=(0.5, 1.5)),
    #dict(type='Resize', scale=(650, 200)), # w, h 
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(960,544), keep_ratio=True),
    #dict(type='StereoResize2', img_scale=(width,height), keep_ratio=True, ratio_range=(0.9, 1.1)),
    dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    #dict(type='Augmentor', image_height=384,
    #                        image_width=512,
    #                        max_disp=256,
    #                        scale_min=0.6,
    #                        scale_max=1.0,
    #                        seed=0,),
    #dict(type='HorizontalFlip', mono=False),
    dict(type='PhotoMetricDistortion'),
    #dict(type='StereoRandomCrop',size=(400,256)),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',      ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',  ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',  ])
]
val_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(512,320), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(384,1280), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs',  'left_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'left_disps',   ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]
test_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'),
    dict(type='LoadAnnotations', views=["left", ],  modalities=["disp"],),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(512,320), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(384,1280), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs',  'left_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'left_disps',   ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=train_annfiles,
        data_prefix=data_prefix,
        eval_modality="disp",
        depth_scale_ratio=256,
        #end_id=10,
        test_mode= False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=test_annfiles,
        #end_id=100,
        data_prefix=data_prefix,
        eval_modality="disp",
        depth_scale_ratio=256,
        test_mode= True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=test_annfiles,
        end_id=40,
        data_prefix=data_prefix,
        eval_modality="disp",
        depth_scale_ratio=256,
        test_mode= True,
        pipeline=test_pipeline))