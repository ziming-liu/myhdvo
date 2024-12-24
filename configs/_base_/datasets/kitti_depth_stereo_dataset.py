import os.path as osp

batch_size = 2
width,height = (800,256)
# dataset settings
dataset_type = 'KITTI-2015'
# data_root = 'datasets/{}/'.format(dataset_type)
# annfile_root = osp.join(data_root, 'annotations')

# root = '/home/youmin/'
root = '/home/ziliu/DenseMatchingBenchmark'
root2 = "/data/acentauri/user/ziliu/data"

data_root = osp.join(root2, 'stereo_matching_data/StereoMatching/', dataset_type)
annfile_root = osp.join(root2, "depth_splits/eigen_monodepth")

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
vis_data_root = osp.join(root2, 'stereo_matching_data/visualization_data/', dataset_type)
vis_annfile_root = osp.join(vis_data_root, 'annotations')

dataset_type = 'KittiDepthStereoDataset'
data_root = '/data/acentauri/user/ziliu/data/'
#data_root_val = 'data/kinetics400/rawframes_val'
#ann_file_train = '/home/ziliu/DeepVO-pytorch/KITTI/base_train_more.pickle'
#ann_file_val = '/home/ziliu/DeepVO-pytorch/KITTI/base_val.pickle'
#ann_file_test = '/home/ziliu/DeepVO-pytorch/KITTI/base_test.pickle'
img_norm_cfg = dict(mean=[88.78708011161852, 93.43778497818349, 91.33551888646076], std=[80.93941240862273, 81.55742718042109, 82.55097977909143], to_rgb=False)

train_pipeline = [
    dict(type='StereoDefinedFrameDecode',dataset="kittidepth", stereo_id=['image_02/data','image_03/data',]), #'proj_depth/groundtruth/image_02','proj_depth/groundtruth/image_03']),
    #dict(type='VOSampleFrames', clip_len=2, frame_interval=1, num_clips=1,),
    #dict(type='StereoRawFrameDecode',  stereo_id=['image_2', 'image_3' ]),
    #dict(type='Resize', scale=(650, 200)), # w, h 
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    #dict(type='StereoResize', scale=(1152,256), keep_ratio=False),
    dict(type='StereoResize', scale=(width,height), keep_ratio=False),
    #dict(type='StereoRandomCrop',size=(400,256)),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs', ] ),
    dict(type='Collect', keys=['left_imgs','right_imgs',   'intrinsics', 'focal', 'baseline','stereo_pose', "pose"], meta_keys=['frame_dir', ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'intrinsics', 'focal', 'baseline','stereo_pose', "pose"])
]
val_pipeline = [
    dict(type='StereoDefinedFrameDecode',dataset="kittidepth", stereo_id=['image_02/data','image_03/data',]),#'proj_depth/groundtruth/image_02','proj_depth/groundtruth/image_03']),
    dict(type='StereoResize', scale=(1280,384), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs', ] ),
    dict(type='Collect', keys=['left_imgs','right_imgs',  'intrinsics', 'focal', 'baseline','stereo_pose', ], meta_keys=['frame_dir',]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'intrinsics', 'focal', 'baseline','stereo_pose', ])
]
test_pipeline = [
    dict(type='StereoDefinedFrameDecode',dataset="kittidepth", stereo_id=['image_02/data','image_03/data',]),#'proj_depth/groundtruth/image_02','proj_depth/groundtruth/image_03']),
    #dict(type='VOSampleFrames', clip_len=1, frame_interval=1, num_clips=1,),
    #dict(type='StereoRawFrameDecode',  stereo_id=['image_2', 'image_3' ]),
    #dict(type='StereoCenterCrop', crop_size=(0.40810811,0.99189189,0.03594771, 0.96405229)),
    dict(type='StereoResize', scale=(1280,384), keep_ratio=False),
    #dict(type='ThreeCrop', crop_size=256),
    #dict(type='Flip', flip_ratio=0),
    dict(type='StereoNormalize', **img_norm_cfg),
    dict(type='StereoFormatShape', input_format='NTCHW', keys=['left_imgs','right_imgs', ] ),
    dict(type='Collect', keys=['left_imgs','right_imgs',   'intrinsics', 'focal', 'baseline','stereo_pose',], meta_keys=['frame_dir', ]),
    dict(type='ToTensor', keys=['left_imgs','right_imgs',  'intrinsics', 'focal', 'baseline','stereo_pose', ])
]

data = dict(
    sparse=True,
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "eigen_train_files.txt"),
        data_prefix=data_root,
        overlap=1,
        frame_shifts=[[-1,0], [0,1], ], # shift steps from given frame of annotions. 
        seq_len_range=[2,2],
        test_mode= False,
        #folder_list=['00', '02', '08', '09'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "eigen_test_files.txt"),
        overlap=1,
        seq_len_range=[1,1],
        end_id=100,
        #folder_list=['03', '04', '05', '06', '07', '10'],
        frame_shifts=[[0,],],
        data_prefix=data_root,
        test_mode= True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=osp.join(annfile_root, "eigen_test_files.txt"),
        overlap=1,
        end_id=100,
        seq_len_range=[1,1],
        frame_shifts=[[0,],],
        #folder_list=['03', '04', '05', '06', '07', '10'],
        data_prefix=data_root,
        test_mode= True,
        pipeline=test_pipeline))