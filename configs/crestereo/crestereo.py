'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 18:29:53
LastEditors: Ziming Liu
LastEditTime: 2023-03-15 00:58:44
Describe: To be same with crestereo original code, we do not use normlization in the data pipeline, 
           images are normed in model/stereo_predictor/crestereo.py
'''

import os.path as osp
_base_ = [
    '../_base_/models/crestereo.py',
    '../_base_/datasets/crestereo_random384x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/crestereo.py'
]

batch_size = 5
root2 = "/home/ziliu/mydata"

data_root = osp.join(root2, 'crestereo')
annfile_root = osp.join(root2, "crestereo")

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
#vis_data_root = osp.join(root2, 'stereo_matching_data/visualization_data/', dataset_type)
#vis_annfile_root = osp.join(vis_data_root, 'annotations')

dataset_type = 'CREStereoDataset'
#data_root = '/data/acentauri/user/ziliu/data/'
#data_root_val = 'data/kinetics400/rawframes_val'
#ann_file_train = '/home/ziliu/DeepVO-pytorch/KITTI/base_train_more.pickle'
#ann_file_val = '/home/ziliu/DeepVO-pytorch/KITTI/base_val.pickle'
#ann_file_test = '/home/ziliu/DeepVO-pytorch/KITTI/base_test.pickle'
train_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True),
    dict(type='LoadAnnotations', views=["left", "right"],  modalities=["disp"]),
    #dict(type='Resize', scale=(650, 200)), # w, h 
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(width,height), keep_ratio=False),
    #dict(type='StereoResize2', img_scale=(width,height), keep_ratio=True, ratio_range=(0.9, 1.1)),
    #dict(type='StereoRandomCrop2', crop_size=crop_size, zeros_disp_max_ratio=1, random_shift=False),
    dict(type='Augmentor', image_height=384,
                            image_width=512,
                            max_disp=256,
                            scale_min=0.6,
                            scale_max=1.0,
                            seed=0,),
    #dict(type='HorizontalFlip', mono=False),
    #dict(type='PhotoMetricDistortion'),
    #dict(type='StereoRandomCrop',size=(400,256)),
    #dict(type='Flip', flip_ratio=0),
    #dict(type='StereoNormalize', **img_norm_cfg),
    #dict(type='StereoPad', size=crop_size, pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps', 'right_disps'    ], meta_keys=[  ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps', 'right_disps'       ])
]
val_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',to_rgb=True),
    dict(type='LoadAnnotations', views=["left", "right"],  modalities=["disp"],),
    #dict(type='StereoResize', scale=(960,544), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    #dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(576,960), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps',    ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs', 'left_disps',    ])
]
test_pipeline = [
    dict(type='LoadStereoImages', to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', to_rgb=True),
    dict(type='LoadAnnotations', views=["left", "right"],  modalities=["disp"],),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(512,320), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    #dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoPad', size=(576,960), pad_val=0, disp_pad_val=0),
    dict(type='StereoFormatShape', input_format='NCHW', keys=['left_imgs','right_imgs', 'left_disps',  'right_disps',   ]),
    dict(type='Collect', keys=['left_imgs','right_imgs', 'left_disps', 'right_disps'   ], meta_keys=[ ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'left_disps',  'right_disps'  ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "crestereo_full_train.json"),
        data_prefix=data_root,
        eval_modality="disparity",
        depth_scale_ratio=32,
        #end_id=50,
        test_mode= False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file="/home/ziliu/mydata/sceneflow/annotations/cleanpass_test.json",
        end_id=100,
        data_prefix="/home/ziliu/mydata/sceneflow",
        eval_modality="disparity",
        depth_scale_ratio=32,
        test_mode= True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file="/home/ziliu/mydata/sceneflow/annotations/cleanpass_test.json",
        end_id=100,
        data_prefix="/home/ziliu/mydata/sceneflow",
        eval_modality="disparity",
        depth_scale_ratio=32,
        test_mode= True,
        pipeline=test_pipeline))

work_dir = "work_dirs/crestereo_base"

resume_from = "work_dirs/crestereo_base/epoch_30.pth"

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

minibatch=500 