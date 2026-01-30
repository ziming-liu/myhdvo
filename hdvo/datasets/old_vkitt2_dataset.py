import os.path as osp

import os
import glob
import cv2
import torch
from torch.nn import functional as F
import pandas as pd
import  time
import numpy as np
import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from zimingvo.utils import register_module_hooks, get_root_logger

from ..core import (mean_average_precision, mean_class_accuracy,rmse,
                    rel,relsqr, log10, rmsedepth, rmsedepthlog, correct,
                    mmit_mean_average_precision, top_k_accuracy)

from ..core.geometry.camera_modules import Intrinsics

from .base import BaseDataset
from .registry import DATASETS
from ..models.geometry import *
import sys
sys.path.append(" /home/ziliu/stereoVOcode/mmvo/")
from KITTI_odometry_evaluation_tool.evaluation import kittiOdomEval

def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def load_vkitti2_odom_intrinsics(camera_intrinsic_path, new_h, new_w):
    """Load virtual kitti2 odometry data intrinscis
    frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
    Args:
        camera_intrinsic_path (str): txt file path
    
    Returns:
        intrinsics (dict): each element contains [cx, cy, fx, fy]
    """
    assert new_h < new_w
    raw_img_h = 375.0 # 1242 x 375
    raw_img_w = 1242.0
    intrinsics = {} # only save intri from left camera, because they are the same.
    with open(camera_intrinsic_path, 'r') as cf:
        raw_intrisic = cf.readlines()
        raw_intrisic = raw_intrisic[1:] # remove heading in txt
        left_cam_intri = raw_intrisic[::2]
        right_cam_intri = raw_intrisic[1::2]
        assert len(left_cam_intri) == len(right_cam_intri)
        for i, item in enumerate(left_cam_intri):
            line_split = [float(value) for value in item.strip().split(' ')[2:] ]
            assert len(line_split) ==4
            intrinsics[i] = [
                            line_split[2]/raw_img_w*new_w, # cx 
                            line_split[3]/raw_img_h*new_h, # cy
                            line_split[0]/raw_img_w*new_w, # fx
                            line_split[1]/raw_img_h*new_h, # fy
                            ] 
    return intrinsics

@DATASETS.register_module()
class VKITTI2StereoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """
    __SCENES__ = ['Scene01', 'Scene02', 'Scene18', 'Scene06', 'Scene20',]
    #__VARIATIONS__ = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 
    #                'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
    __VARIATIONS__ = ['morning' #'15-deg-left', #'15-deg-right', '30-deg-left', '30-deg-right', 
                   # 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'
                    ]

    def __init__(self,  pipeline, 
                 seq_len_range, 
                 ann_file=None,
                 test_seq_id=None,
                 scene_list=None,
                 overlap=1, 
                 end_id=None,
                 data_prefix=None,
                 pred_depth_dir_left=None,
                 pred_depth_dir_right=None,
                 scale=(848,256),
                 depth_range=(1, 655.35), 
                 disparity_range=(1, 192),
                 stereo_id=["Camera_0","Camera_1"], 
                 test_mode=False,
                 filename_tmpl='rgb_{:0>5}.jpg', 
                 depth_filename_tmpl='depth_{:0>5}.png', 
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 modality='RGB',
                 split=None,
                 sample_by_class=False,
                 power=None,
                 regression=True,
                 sample_times=1, 
                 pad_y=False, 
                 gtmask_generator=False,
                  **kwargs):
        self.test_mode = test_mode
        self.depth_range = depth_range
        self.disparity_range = disparity_range
        self.end_id = end_id
        self.data_prefix = data_prefix
        self.gtmask_generator = gtmask_generator
        self.pose_dirs = []
        self.seq_dirs = []
        if split is None:
            if self.test_mode:
                print("INFO test sequence {}".format(test_seq_id))
                self.scene_list =[ test_seq_id ]
            else: # training
                if scene_list is None:
                    print("INFO scene_list is None")
                    self.scene_list = self.__SCENES__#[:2]
                else:
                    print("INFO using scene_list {} for training ".format(scene_list))
                    self.scene_list = scene_list

        elif split == "split1":
            if self.test_mode:
                print("INFO test sequence {}".format(test_seq_id))
                self.scene_list =[ test_seq_id ]
            else: # training
                if scene_list is None:
                    print("INFO scene_list is None")
                    self.scene_list = self.__SCENES__[:4]
                else:
                    print("INFO using scene_list {} for training ".format(scene_list))
                    self.scene_list = scene_list
        for scene_id in self.scene_list:    
            for variation_id in self.__VARIATIONS__:
                print(data_prefix)
                print(scene_id)
                print(variation_id)
                print(osp.join(data_prefix, scene_id, variation_id, "extrinsic.txt"))
                self.pose_dirs.append(osp.join(data_prefix, scene_id, variation_id, "extrinsic.txt"))
                self.seq_dirs.append(osp.join(data_prefix, scene_id, variation_id,))
        #self.pose_dir = osp.join(data_prefix, 'pose_GT')
        #self.seq_dir = osp.join(data_prefix, 'sequences')
        self.pred_depth_dir_left = pred_depth_dir_left
        self.pred_depth_dir_right = pred_depth_dir_right
        self.logger = get_root_logger()
        self.stereo_id = stereo_id
        #self.folder_list = folder_list
        self.seq_len_range = seq_len_range
        self.overlap = overlap
        self.sample_times = sample_times
        self.pad_y = pad_y
        #self.scale = scale
        self.ann_file = ann_file
        #self.video_infos = self.load_annotations()
        #print(len(self.video_infos[1000]['pose']))
        #exit()
        #print(" len of video infos -> {}".format(len(self.video_infos)))
        #self.seq_len_list = list(self.video_infos.seq_len)
        #self.image_arr = np.asarray(self.video_infos.frame_paths)  # image paths
        #self.groundtruth_arr = np.asarray(self.video_infos.pose)
        self.regression = regression
        #self.start_index = start_index
        self.filename_tmpl = filename_tmpl
        self.depth_filename_tmpl = depth_filename_tmpl
        self.with_offset = with_offset
        self.origin_size = [] # include N sequences's image/depth size
        self.video_infos = self.load_annotations()
        super().__init__(ann_file,
                 pipeline,
                 data_prefix=data_prefix,
                 test_mode=test_mode,
                 multi_class=multi_class,
                 num_classes=num_classes,
                 modality=modality,
                 sample_by_class=sample_by_class,
                 power=power, **kwargs)


    #def get_intrinsics_param(self, calib_path):
    #    """Read intrinsics parameters for each dataset
    #    
    #    Returns:
    #        intrinsics_param (list): [cx, cy, fx, fy]
    #    """
    #    assert self.scale[1] < self.scale[0]
    #    # becuse the camera intrincis is constant, don't need to return each frame's intri
    #    # only return one intricis (left camera)
    #    intrinsics_param = load_vkitti2_odom_intrinsics(
    #                    calib_path,
    #                    self.scale[1], self.scale[0] # h,w
    #                    )[0]
    #    return intrinsics_param
    
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        
            
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        #results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl
        results['depth_filename_tmpl'] = self.depth_filename_tmpl
        results['pose'] = self._process_gt_pose(results['pose'])
        results['right_pose'] = self._process_gt_pose(results['right_pose'])

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        if self.pred_depth_dir_left is not None:
            results['pred_depth_dir_left'] = self.pred_depth_dir_left
        if self.pred_depth_dir_right is not None:
            results['pred_depth_dir_right'] = self.pred_depth_dir_right
        results['modality'] = self.modality
        #results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl
        results['depth_filename_tmpl'] = self.depth_filename_tmpl
        results['pose'] = self._process_gt_pose(results['pose'])
        results['right_pose'] = self._process_gt_pose(results['right_pose'])
        return self.pipeline(results)
    
    def _process_gt_pose(self, gt_pose):
        """
        gt_pose: absolute pose w.r.t the world coordinate

        return relative_pose: 
        """
        # processing GT pose
        T_ref, T_tar = gt_pose
        T_ref, T_tar = torch.from_numpy(T_ref), torch.from_numpy(T_tar)
        #print("T tar")
        #print(T_tar)
        #print("T tar .t")
        #print(torch.linalg.inv(T_tar))
        #print(T_tar)
        #print("T ref")
        #print(T_ref)
        #inv_T_tar = torch.linalg.inv(T_tar)
        # the T_tar == cTw   T_ref == rTw
        relatvie_cTr = torch.mm(T_tar.float(), torch.linalg.inv(T_ref.float())).float() # cTr
        relatvie_rTc = torch.mm(T_ref.float(), torch.linalg.inv(T_tar.float())).float() # rTc
        #print("relative pose")
        #print(relatvie_pose)
        #exit()
        # print('Item after transform: ' + str(index) + '   ' + str(groundtruth_sequence))
        relative_pose = torch.stack([relatvie_cTr,relatvie_rTc])
        return relative_pose
    def load_vkitti2_odom_intrinsics(self, camera_intrinsic_path):
        """Load virtual kitti2 odometry data intrinscis
        frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        Args:
            camera_intrinsic_path (str): txt file path
        
        Returns:
            intrinsics (dict): each element contains [cx, cy, fx, fy]
        """
        #assert new_h < new_w
        #raw_img_h = 375.0 # 1242 x 375
        #raw_img_w = 1242.0
        left_cam_intrinsics = None # only save intri from left camera, because they are the same.
        right_cam_intrinsics = None
        with open(camera_intrinsic_path, 'r') as cf:
            raw_intrisic = cf.readlines()
            raw_intrisic = raw_intrisic[1:] # remove heading in txt
            left_cam_intri = raw_intrisic[::2]
            right_cam_intri = raw_intrisic[1::2]
            assert len(left_cam_intri) == len(right_cam_intri)
            #for i, item in enumerate(left_cam_intri):
            # left 
            line_split = [float(value) for value in left_cam_intri[0].strip().split(' ')[2:] ]
            assert len(line_split) ==4
            left_cam_intrinsics = [
                            line_split[2],#/raw_img_w*new_w, # cx 
                            line_split[3],#/raw_img_h*new_h, # cy
                            line_split[0],#/raw_img_w*new_w, # fx
                            line_split[1],#/raw_img_h*new_h, # fy
                            ] 
            # right
            line_split = [float(value) for value in right_cam_intri[0].strip().split(' ')[2:] ]
            assert len(line_split) ==4
            right_cam_intrinsics = [
                            line_split[2],#/raw_img_w*new_w, # cx 
                            line_split[3],#/raw_img_h*new_h, # cy
                            line_split[0],#/raw_img_w*new_w, # fx
                            line_split[1],#/raw_img_h*new_h, # fy
                            ] 
            assert left_cam_intrinsics == right_cam_intrinsics,"left right cameras should have same intrinsics"
        
        cx, cy, fx, fy = left_cam_intrinsics
        left_cam_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])
        cx, cy, fx, fy = right_cam_intrinsics
        right_cam_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])

        def proc_K(K):
            K44 = np.eye(4)
            K44[:3, :3] = K 

            #invK44 = np.eye(4)
            #invK44[:3, :3] = K
            #invK44 = np.linalg.inv(invK44)
            return K44#,invK44
        left_K44 = proc_K(left_cam_intrinsics)
        right_K44 = proc_K(right_cam_intrinsics)
        output_cam_intrinsics = np.stack( (left_K44, right_K44) )
        return output_cam_intrinsics

    def load_annotations(self,):

        #if os.path.exists(self.ann_file):
        #    return pd.read_pickle(self.ann_file)['video_infos']
        X_rgb_path_left, X_rgb_path_right, Y_pose_abs = [], [], []
        right_Y_pose_abs = []
        X_depth_path_left, X_depth_path_right = [], []
        Y_stereo_pose_abs = []
        X_len = []
        X_dir = []
        X_focal=[]
        X_baseline=[]
        X_intrinsics = []
        X_start_ind = []
        X_total_frames = []
        video_infos = []
        print( " total {} # seqs will be loaded".format(len(self.seq_dirs)))
        for seq_i in range(len(self.seq_dirs)):
            start_t = time.time()
            num_frames = -1
            pose_path = self.pose_dirs[seq_i]
            # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
            with open(pose_path, 'r') as pf:
                raw_poses = pf.readlines()
                raw_poses = raw_poses[1:] # remove headinge
                raw_poses = [np.array([ float(value) for value in pose_item.strip().split(' ')[2:]]).reshape((4,4)) for pose_i, pose_item in enumerate(raw_poses)]
                left_poses_abs = raw_poses[0::2]
                right_poses_abs = raw_poses[1::2]
                num_frames = len(left_poses_abs)
                print("lines {}".format(len(raw_poses)))
                print("num poses: l {} r {}".format(len(left_poses_abs), len(right_poses_abs)))
                assert len(left_poses_abs) == len(right_poses_abs), "left camera and right camera should correspond"
            # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
            camera_intrinsic_path = os.path.join(self.seq_dirs[seq_i], "intrinsic.txt") # 
            cam_intrinsics = self.load_vkitti2_odom_intrinsics(camera_intrinsic_path) # a dict of camera 0, 1, 2, 3, tr 
            focal =  cam_intrinsics[0][0][0] #725.0087  fx
            baseline = 0.532725 # baseline of the left came relative to the right cam is fixed as this value
            #cam_intrinsics = intrinsics # only save one for each sequence, cx cy fx fy
            # frame rgb load
            f_rgb_paths_left = glob.glob(os.path.join(self.seq_dirs[seq_i],"frames/rgb/Camera_0","*.jpg"))
            f_rgb_paths_right = glob.glob(os.path.join(self.seq_dirs[seq_i],"frames/rgb/Camera_1","*.jpg"))
            f_rgb_paths_left.sort()
            f_rgb_paths_right.sort()
            assert len(f_rgb_paths_left) == num_frames
            assert len(f_rgb_paths_right) == num_frames
            #print(" promise the f_rgb_paths_left/right is ordered: \n {} \n {}".format(f_rgb_paths_left,f_rgb_paths_right))
            # read heigh and width
            
            example_sam = cv2.imread(f_rgb_paths_left[-1])
            H,W,C = example_sam.shape
            

            # frame depth load
            f_depth_paths_left = glob.glob(os.path.join(self.seq_dirs[seq_i],"frames/depth/Camera_0","*.png"))
            f_depth_paths_right = glob.glob(os.path.join(self.seq_dirs[seq_i],"frames/depth/Camera_1","*.png"))
            f_depth_paths_left.sort()
            f_depth_paths_right.sort()
            assert len(f_depth_paths_left) == num_frames
            assert len(f_depth_paths_right) == num_frames
            #print(" promise the f_depth_paths_left/right is ordered: \n {} \n {}".format(f_depth_paths_left,f_depth_paths_right))
            

            # Fixed seq_len
            if self.seq_len_range[0] == self.seq_len_range[1]:
                if self.sample_times > 1:
                    # sample a same seq multi times 
                    sample_interval = int(np.ceil(self.seq_len_range[0] / self.sample_times))
                    start_frames = list(range(0, self.seq_len_range[0], sample_interval))
                    print('Samples start from frame: {}'.format(start_frames))
                else:
                    start_frames = [0,]
                    print('Samples start from frame: 0')

                for st in start_frames: # default once only
                    # rest_frames: the rest number of frames since start frame
                    seq_len = self.seq_len_range[0]
                    rest_frames = len(f_rgb_paths_left) - st
                    jump = seq_len - self.overlap
                    #res = rest_frames % seq_len
                    res = rest_frames % jump
                    if res != 0:
                        rest_frames = rest_frames - res
                    
                    for i in range(st, rest_frames, jump):
                        if i+seq_len >rest_frames:
                            continue
                        X_rgb_path_left.append(f_rgb_paths_left[i:i+seq_len])
                        #if len(self.stereo_id) ==2:
                        X_rgb_path_right.append(f_rgb_paths_right[i:i+seq_len])
                        # add depth
                        X_depth_path_left.append(f_depth_paths_left[i:i+seq_len])
                        #if len(self.stereo_id) ==2:
                        X_depth_path_right.append(f_depth_paths_right[i:i+seq_len])

                        stereo_T = np.eye(4, dtype=np.float32)
                        #baseline_sign = -1 if do_flip else 1
                        baseline_sign = baseline #1 
                        side_sign = -1 # if side == "l" else 1
                        stereo_T[0, 3] = side_sign * baseline_sign #* 0.1 # we already make the baseline to be (m)

                        rTl = stereo_T # torch.from_numpy(stereo_T)

                        stereo_T = np.eye(4, dtype=np.float32)
                        #baseline_sign = -1 if do_flip else 1
                        baseline_sign = baseline #1 
                        side_sign = 1  # side==right  for sparse warp
                        stereo_T[0, 3] = side_sign * baseline_sign #* 0.1 # we already make the baseline to be (m)

                        lTr = stereo_T # torch.from_numpy(stereo_T)
                        stereo_pose = np.stack((rTl, lTr))
                        
                        Y_stereo_pose_abs.append(stereo_pose)
                        Y_pose_abs.append(left_poses_abs[i:i+seq_len])
                        right_Y_pose_abs.append(right_poses_abs[i:i+seq_len])
                        X_len.append(seq_len)
                        X_dir.append(self.seq_dirs[seq_i]) 
                        X_start_ind.append(int(f_rgb_paths_left[i].split('/')[-1].split('.')[0].split('_')[-1]))
                        X_total_frames.append(num_frames)
                        X_intrinsics.append(cam_intrinsics)
                        X_focal.append(focal)
                        X_baseline.append(baseline)
                        self.origin_size.append((H,W))
                        #if self.gtmask_generator and not self.test_mode:
                        #    self.gt_mask_generator(f_rgb_paths_left[i:i+seq_len], f_rgb_paths_right[i:i+seq_len], f_depth_paths_left[i:i+seq_len], f_depth_paths_right[i:i+seq_len], \
                        #        [left_poses_abs[i], left_poses_abs[i+seq_len-1]], [right_poses_abs[i], right_poses_abs[i+seq_len-1]], focal, baseline, cam_intrinsics,stereo_pose )
            
            print('Seq: {} is sampeled in {} sec'.format(self.seq_dirs[seq_i], time.time()-start_t))
        
        assert len(X_rgb_path_left) == len(X_depth_path_left)
        assert len(X_len) == len(Y_pose_abs)
        assert len(Y_pose_abs) == len(X_rgb_path_left)
        for i in range(len(Y_pose_abs)):
            video_infos.append(dict(total_frames=X_total_frames[i], seq_len=X_len[i],
                                    original_size = self.origin_size[i],
                                    frame_dir=X_dir[i],
                                    depth_dir=X_dir[i],
                                    intrinsics = X_intrinsics[i],
                                    focal = X_focal[i],
                                    baseline = X_baseline[i],
                                    stereo_pose = Y_stereo_pose_abs[i],
                                    start_index=X_start_ind[i],
                                    frame_paths=[X_rgb_path_left[i], X_rgb_path_right[i]] if len(self.stereo_id)==2 \
                                        else [X_rgb_path_left[i], ],
                                    depth_paths=[X_depth_path_left[i], X_depth_path_right[i]] if len(self.stereo_id)==2 \
                                        else [X_depth_path_left[i], ],
                                    pose=[Y_pose_abs[i][0], Y_pose_abs[i][-1]],
                                    right_pose= [right_Y_pose_abs[i][0], right_Y_pose_abs[i][-1]])) # poses for each frames in sequence i
        print(f"total len {len(video_infos)}")
        if self.end_id is not None:
            return video_infos[:self.end_id]
        return video_infos

    def evaluate(self,
                 results,
                 eval_tasks=['depth',],
                 metrics=['rel','relsqr','log10','rmsedepth','rmsedepthlog','correct'],
                 metric_options=None,
                 vis=False,
                 logger=None,
                 **deprecated_kwargs):#metric_options=dict(rmse=dict(items=['translation', 'rotation'])),
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/zimingvo/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)
        print("evaluating...")
        

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/zimingvo/pull/286' "  # noqa: E501
                'for more details')

        depth_eval_results = OrderedDict()
        if "depth" in eval_tasks:
            

            folder_list = []
            for a in self.scene_list: # keep test or train mod correct
                for b in self.__VARIATIONS__:
                    folder_list.append(a+'/'+b)

            all_pred_depth = results[0] # batch views h w 
            all_gt_depth = results[1] # batch views h w 
            all_multimasks = results[2]
            with_mask = True
            if len(all_multimasks) == 0: with_mask = False
            B = len(all_pred_depth)
            V, H, W = all_pred_depth[0].shape
            if V==1: views = ["left"]
            elif V==2: views = ["left", "right"]
            else: raise ValueError
            for idx, view in enumerate(views):
                pred_depth = [all_pred_depth[k][idx, :, :] for k in range(B)]
                gt_depth = [all_gt_depth[k][idx, :, :] for k in range(B)]
                multimask = [all_multimasks[k][idx, :, :] for k in range(B)] if with_mask == True else None
            
                #print("pred_depth 80m\n {}".format(pred_depth[10]))
                num_samples = len(pred_depth)
                min_depth = self.depth_range[0] #2.0116
                max_depth =  self.depth_range[1]  #655.35 #80.0 # sky = 655.35 filter the sky regions
                #print("pred_depth clamp 80m \n {}".format(pred_depth[10]))
                rms     = np.zeros(num_samples, np.float32)
                log_rms = np.zeros(num_samples, np.float32)
                abs_rel = np.zeros(num_samples, np.float32)
                sq_rel  = np.zeros(num_samples, np.float32)
                d1_all  = np.zeros(num_samples, np.float32)
                a1      = np.zeros(num_samples, np.float32)
                a2      = np.zeros(num_samples, np.float32)
                a3      = np.zeros(num_samples, np.float32)
                def compute_errors(gt, pred):
                    #print("pred\n {}".format(pred))
                    #print("gt \n {}".format(gt))
                    thresh = np.maximum((gt / pred), (pred / gt))
                    a1 = (thresh < 1.25   ).mean()
                    a2 = (thresh < (1.25 ** 2)).mean()
                    a3 = (thresh < (1.25 ** 3)).mean()

                    rmse = (gt - pred) ** 2
                    rmse = np.sqrt(rmse.mean())

                    rmse_log = (np.log(gt) - np.log(pred)) ** 2
                    rmse_log = np.sqrt(rmse_log.mean())

                    abs_rel = np.mean(np.abs(gt - pred) / gt)

                    sq_rel = np.mean((np.abs(gt - pred)**2) / gt)

                    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
                #print("pred_depth clamp 80m \n {}".format(pred_depth[10]))
                
                masked_percent_depth_range = 0
                masked_percent_crop = 0
                masked_percent_stereo = 0
                masked_percent_dr_stereo = 0
                masked_percent_temporal = 0
                masked_percent_dr_temporal = 0
                num_samples = len(gt_depth)

                for i in range(len(gt_depth)):
                    try:
                        _, gt_height, gt_width = gt_depth[i].shape
                    except:
                        gt_height, gt_width = gt_depth[i].shape
                    assert gt_height >1 
                    assert gt_width >1
                    total_pixels = gt_height*gt_width
                    # follow the operations in https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/utils/evaluate_kitti.py
                    mask0 = (pred_depth[i] < 999)
                    #print(mask0)
                    pred_depth[i] = np.clip(pred_depth[i], min_depth, max_depth)
                    mask = np.logical_and(gt_depth[i] >min_depth, gt_depth[i] < max_depth)
                    mask = np.logical_and(mask0, mask)
                    masked_percent_depth_range += np.sum(mask) /total_pixels
                    garg_crop = True # vkitti2 don't use kitti crop. 
                    eigen_crop = False
                    if garg_crop or eigen_crop:
                        
                        # crop used by Garg ECCV16
                        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                        if garg_crop:
                            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                            0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                        # crop we found by trial and error to reproduce Eigen NIPS14 results
                        elif eigen_crop:
                            crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                            0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                        crop_mask = np.zeros(mask.shape)
                        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                        masked_percent_crop += np.sum(crop_mask) / total_pixels
                        mask = np.logical_and(mask, crop_mask)
                    #masks.append(np.logical_and(gt_depth[i]>min_depth, gt_depth[i]<max_depth))
                    #print(pred_depth[i].shape)
                    #mask_max = gt_depth[i]>max_depth
                    #mask_min = gt_depth[i]<min_depth
                    #gt_depth[i][mask_max] = max_depth
                    #pred_depth[i][mask_max] = max_depth
                    #gt_depth[i][mask_min] = min_depth
                    #pred_depth[i][mask_min] = min_depth
                    abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[i][mask], pred_depth[i][mask])
                    

                print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
                print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),  a1.mean(), a2.mean(), a3.mean()))
                eval_res = {'masked_percent_depth_range':masked_percent_depth_range/num_samples, 'masked_percent_crop':masked_percent_crop/num_samples, 'abs_rel':abs_rel.mean(), 'sq_rel': sq_rel.mean(), 'rms':rms.mean(), 'log_rms': log_rms.mean(), 'd1_all':d1_all.mean(), 'a1':a1.mean(), 'a2':a2.mean(), 'a3': a3.mean()}
                log_msg = []
                log_msg.append(f'\n ##View:{view} Std evaluation results \n')
                for k, acc in eval_res.items():
                    #depth_eval_results[f'_{k}'] = acc
                    log_msg.append(f'  {k}: {acc:.4f}  ')
                    self.logger.info(f'  {k}: {acc:.4f}  ')
                log_msg = ' '.join(log_msg)
                print_log(log_msg, logger=logger)
            
                # with masked 
                if with_mask:
                    for i in range(len(gt_depth)):
                        # follow the operations in https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/utils/evaluate_kitti.py
                        pred_depth[i] = np.clip(pred_depth[i], min_depth, max_depth)
                        mask = np.logical_and(gt_depth[i] >min_depth, gt_depth[i] < max_depth)
                        garg_crop = False 
                        eigen_crop = False
                        if garg_crop or eigen_crop:
                            assert gt_height >1 
                            assert gt_width >1
                            # crop used by Garg ECCV16
                            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                            if garg_crop:
                                crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                                0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                            # crop we found by trial and error to reproduce Eigen NIPS14 results
                            elif eigen_crop:
                                crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                                0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                            crop_mask = np.zeros(mask.shape)
                            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                            mask = np.logical_and(mask, crop_mask)
                        
                        mask = np.logical_and(mask, multimask[i])
                        masked_percent_stereo += np.sum(multimask[i]) / total_pixels
                        masked_percent_dr_stereo += np.sum(mask) / total_pixels
                        #masks.append(np.logical_and(gt_depth[i]>min_depth, gt_depth[i]<max_depth))
                        #print(pred_depth[i].shape)
                        mask_max = gt_depth[i]>max_depth
                        mask_min = gt_depth[i]<min_depth
                        #gt_depth[i][mask_max] = max_depth
                        #pred_depth[i][mask_max] = max_depth
                        #gt_depth[i][mask_min] = min_depth
                        #pred_depth[i][mask_min] = min_depth
                        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[i][mask], pred_depth[i][mask])
                        

                    #print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
                    #print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
                    #eval_res = {'masked_percent_stereo':masked_percent_stereo/num_samples, 'masked_percent_dr_stereo':masked_percent_dr_stereo/num_samples, 'abs_rel':abs_rel.mean(), 'sq_rel': sq_rel.mean(), 'rms':rms.mean(), 'log_rms': log_rms.mean(), 'd1_all':d1_all.mean(), 'a1':a1.mean(), 'a2':a2.mean(), 'a3': a3.mean()}
                    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
                    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean()*100, a2.mean()*100, a3.mean()*100))
                    eval_res = {'abs_rel':abs_rel.mean(), 'sq_rel': sq_rel.mean(), 'rms':rms.mean(), 'log_rms': log_rms.mean(), 'd1_all':d1_all.mean(), 'a1':a1.mean()*100, 'a2':a2.mean()*100, 'a3': a3.mean()*100}
                    log_msg = []
                    log_msg.append(f'\n ##View:{view} stereo masked evaluation results \n')
                    for k, acc in eval_res.items():
                        #depth_eval_results[f'_{k}'] = acc
                        log_msg.append(f'  {k}: {acc:.4f}  ')
                        self.logger.info(f'  {k}: {acc:.4f}  ')
                    log_msg = ' '.join(log_msg)
                    print_log(log_msg, logger=logger)
                    
            # the range is 2.0116 - 655.35 for predicted depth 
            #pred_depth = [pred_depth[i] for i in range(len(pred_depth))]
            #gt_depth = [gt_depth[i] for i in range(len(gt_depth))]
            #print("gt depth \n{}".format(gt_depth[0]))
            #print("pred depth \n{}".format(pred_depth[0]))
            #import cv2
            #vis_pred =  cv2.normalize(pred_depth[0].squeeze(0), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            #cv2.imwrite("pred_depth.png", vis_pred)
            #vis_gt =  cv2.normalize(gt_depth[0].squeeze(0), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            #cv2.imwrite("gt_depth.png", vis_gt)
            """
            frame_dir = results[-1]
            # in vkitti2, dir is like 'Scene1/morning'
            for z in range(len(frame_dir)):
                frame_dir[z] = '/'.join(frame_dir[z].split('/')[-2:])
                assert frame_dir[z] in folder_list, "{} and {}".format(frame_dir[z], folder_list)
            assert len(pred_depth) == len(self)
            assert len(gt_depth) == len(self)
            assert len(frame_dir) == len(self)

            for metric in metrics:
                msg = f'Evaluating {metric} ...'
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                res = globals()['{}'.format(metric)](pred_depth, gt_depth, frame_dir, folder_list, )
                log_msg = []
                log_msg.append(f'\n{metric} evaluation results')
                for k, acc in res.items():
                    depth_eval_results[f'{metric}_{k}'] = acc
                    log_msg.append(f'\n{metric}_{k}acc\t{acc:.4f}')
                    
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            """
        
        if "disparity" in eval_tasks:
            from imageio import imread
            epoch_error = 0
            valid_iteration = 0
            three_px_acc_all = 0
            max_disp = 192
            print("eval total size {}".format(len(self.video_infos)))
            for i in range(len(self.video_infos)):
                data = {}
                left_disp_path = self.video_infos[i]["depth_paths"][0][0]
                #right_disp_path = self.video_infos[i]["right_depth_paths"][0]
                left_img_path = self.video_infos[i]["frame_paths"][0][0]
                #right_img_path = self.video_infos[i]["right_frame_paths"][0]
                #print(left_disp_path)
                leftDepth = 0.01 * cv2.imread(left_disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                leftDisp = self.video_infos[i]["focal"] * self.video_infos[i]["baseline"] / leftDepth
                leftDisp = leftDisp[np.newaxis, ...]
                data.update(leftDisp=torch.FloatTensor(leftDisp))
                #rightDisp = imread(
                #        right_disp_path).astype(np.float32) / 256.0
                #rightDisp = rightDisp[np.newaxis, ...]
                #data.update(rightDisp=torch.FloatTensor(rightDisp))
                leftImage = imread(
                     left_img_path
                    ).transpose(2, 0, 1).astype(np.float32)[:3]
                data.update(leftImage=torch.FloatTensor(leftImage).unsqueeze(0))
                #rightImage = imread(
                #    right_img_path
                #    ).transpose(2, 0, 1).astype(np.float32)[:3]
                #data.update(rightImage=torch.FloatTensor(rightImage).unsqueeze(0))
                h,w = leftImage.shape[1],leftImage.shape[2]
                original_size = (h,w)
                data.update({"original_size": original_size})

                # load the predicted disparity 
                pred_disparity = self.video_infos[i]["focal"]*self.video_infos[i]["baseline"]/ torch.FloatTensor(results[0][i])#.unsqueeze(0) # b h w
                
                #result = disp_(deprecated_kwargs["cfg"], pred_disparity, data)
                #filter_result = {}
                #filter_result.update(Error=result['Error'])

                

                target = data["leftDisp"]
                target=torch.squeeze(target,1)
                #print("target >> ", target.shape)
                #_,h,w = target.shape
                #target = target[:,int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w)]
                #pred_disparity = pred_disparity[:,int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w)]

                mask = (target < max_disp)&(target>0) 
                mask.detach_()
                mask = (mask & ((pred_disparity > self.disparity_range[0])&(pred_disparity<self.disparity_range[1]) ))
                mask.detach_()
                valid = target[mask].size()[0]
                # different size 
                #print("pred size {}, target size {} ".format(pred_disparity.shape, target.shape))
                if pred_disparity.shape[-2:] != target.shape[-2:]:
                    #print(pred_disparity.shape)
                    #print(target.shape)
                    scale = target.shape[-1] / pred_disparity.shape[-1]
                    pred_disparity =scale * torch.nn.functional.interpolate(pred_disparity.cuda().unsqueeze(0), target.shape[-2:],mode="bilinear" ).cpu().squeeze(0)


                if valid>0:
                    #print("mask ", mask.shape)
                    #print("pred ", pred_disparity.shape)
                    #print("pred: {}".format(pred_disparity[mask][:60]))
                    #print("target: {}".format(target[mask][:60]))
                    #print("num zeros: {}, non zeros: {}".format(len(target[target==0]),len(target[mask])))
                    error = torch.mean(torch.abs(pred_disparity[mask] - target[mask]))  

                    valid_iteration +=1 
                    epoch_error += error.item()

                    #computing 3-px error#                
                    pred_disp = pred_disparity.cpu().detach()                                                                                                                          
                    true_disp = target.cpu().detach()
                    #print("pred disp {}, true disp {}".format(pred_disp.shape,true_disp.shape))
                    disp_true = true_disp
                    
                    index = np.argwhere(mask) #np.argwhere(true_disp<max_disp)
                    #print("index {}".format(index.shape))
                    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
                    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
                    three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))
                    three_px_acc_all += three_px_acc
    
                    #print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(i, len(self.video_infos), error.item(), three_px_acc))
            
            print("===> Test: Avg. Error: (EPE: {:.4f}, 3PE: {:.4f} %)".format(epoch_error/valid_iteration, 100*three_px_acc_all/valid_iteration))


        pose_eval_results = OrderedDict()
        if "pose" in eval_tasks:
            # eval 
            eval_dict =  {"gt_dir":deprecated_kwargs["gt_pose_dir"], \
                 "result_dir": deprecated_kwargs["pred_pose_dir"],\
                     "eva_seqs": deprecated_kwargs["test_seq_id"][-2:]+"_pred" }
            pose_eval = kittiOdomEval(eval_dict)
            pose_eval.eval(toCameraCoord=False) 

        return tuple([depth_eval_results,pose_eval_results])


    def gt_mask_generator(self, rgb_left_path, rgb_right_path, depth_left_path, depth_right_path, left_pose, right_pose, focal, baseline, intrinsics, stereo_pose):
        import cv2
        from zimingvo.models.visual_odometry.layers import BackprojectDepth,Project3D
        #print(rgb_left_path)
        #print(rgb_right_path)
        ##print(depth_left_path)
        #print(depth_right_path)
        def load_tensor(rgb_left_path, rgb_right_path, depth_left_path, depth_right_path):
            def load_img_tensor(imgpath):
                img = cv2.imread(imgpath)
                img=np.transpose(img,(2,0,1))
                return torch.from_numpy(img).unsqueeze(0).float()
            left_img = load_img_tensor(rgb_left_path)
            right_img = load_img_tensor(rgb_right_path)
            b, c, h, w = left_img.shape
            left_depth = 0.01 * cv2.imread(depth_left_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32).reshape(1,h,w)
            right_depth = 0.01 * cv2.imread(depth_right_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32).reshape(1,h,w)
            left_depth = torch.from_numpy(left_depth).unsqueeze(0).float()
            right_depth = torch.from_numpy(right_depth).unsqueeze(0).float()
            return left_img, right_img, left_depth, right_depth
        t0_left_img, t0_right_img, t0_left_depth, t0_right_depth = load_tensor(rgb_left_path[0], rgb_right_path[0], depth_left_path[0], depth_right_path[0])
        t1_left_img, t1_right_img, t1_left_depth, t1_right_depth = load_tensor(rgb_left_path[1], rgb_right_path[1], depth_left_path[1], depth_right_path[1])
        b, c, h, w = t0_left_img.shape
        #print(t0_left_img.shape)
        relative_pose = self._process_gt_pose(left_pose)
        cTr, rTc = relative_pose[0].unsqueeze(0), relative_pose[1].unsqueeze(0)
        relative_pose_right = self._process_gt_pose(right_pose)
        cTr_right, rTc_right = relative_pose_right[0].unsqueeze(0), relative_pose_right[1].unsqueeze(0)

        intrinsics = torch.from_numpy(intrinsics).float().unsqueeze(0)
        left_K44, right_K44 = intrinsics[:,0,:,:].float(), intrinsics[:,1,:,:].float()
        #left_K44, right_K44 = torch.from_numpy(left_K44), torch.from_numpy(right_K44)
        left_invK44, right_invK44 = torch.linalg.inv(left_K44), torch.linalg.inv(right_K44)
        
        rTl, lTr = torch.from_numpy(stereo_pose).float()

        self.layer_backproject_depth  = BackprojectDepth(batch_size=b, height=h, width=w, device='cpu')
        self.layer_project = Project3D(batch_size=1, height=h, width=w)
        y_base, x_base = torch.meshgrid(
                                torch.linspace(0.0 , h - 1.0 , h),
                                torch.linspace(0.0,   w - 1.0,  w))
        y_base, x_base =  y_base.unsqueeze(0).repeat(b,1,1).long(), \
                         x_base.unsqueeze(0).repeat(b,1,1).long()

        # forward_tamporal mask
        ### on left view 
        cam_points = self.layer_backproject_depth(t1_left_depth, left_invK44) 
        #pix_coordi = self.layer_project(cam_points, left_K44, rTc)
        #I_wcL = F.grid_sample(t0_left_img, pix_coordi, mode='bilinear', padding_mode="zeros")
        P = torch.matmul(left_K44, rTc)[:, :3, :]
        cam_points = torch.matmul(P, cam_points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
        pix_coords = pix_coords.view(b, 2, h, w)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coordi = torch.round(pix_coords)
        x_idx = torch.clamp(pix_coordi[..., 0], 0, w-1).long()
        y_idx =torch.clamp( pix_coordi[..., 1], 0, h-1).long()
        #print("IwcL", I_wcL)
        #print("t1_left_img",t1_left_img)
        #t1_left_gtmask = torch.zeros((b,c,h,w))
        #t1_left_gtmask[torch.abs(I_wcL-t1_left_img)<5] = 255
        #t1_left_gtmask = t1_left_gtmask.mean(-3,False)
        t1_left_gtmask = 255* torch.ones_like(t1_left_depth)
        t1_left_gtmask[t1_left_depth[:,0,y_base,x_base]>t0_left_depth[:,0,y_idx, x_idx]] =0
        save_path = rgb_left_path[1].split('/')
        save_path[-3] = "forward_temporal_mask"
        #print("t1_left_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
       #cv2.imwrite( '/'.join(save_path), np.transpose(I_wcL.squeeze().cpu().numpy(),(1,2,0)))
        cv2.imwrite( '/'.join(save_path), t1_left_gtmask.squeeze().cpu().numpy())

        ### on right view
        cam_points = self.layer_backproject_depth(t1_right_depth, right_invK44) 
        #pix_coordi = self.layer_project(cam_points, right_K44, rTc_right)
        #I_wcR = F.grid_sample(t0_right_img, pix_coordi, mode='bilinear', padding_mode="border")
        #t1_right_gtmask = torch.zeros((b,c,h,w))
        #t1_right_gtmask[torch.abs(I_wcR-t1_right_img)<5] = 255
        #t1_right_gtmask = t1_right_gtmask.mean(-3,False)
        P = torch.matmul(right_K44, rTc_right)[:, :3, :]
        cam_points = torch.matmul(P, cam_points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
        pix_coords = pix_coords.view(b, 2, h, w)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coordi = torch.round(pix_coords)
        x_idx = torch.clamp(pix_coordi[..., 0], 0, w-1).long()
        y_idx =torch.clamp( pix_coordi[..., 1], 0, h-1).long()
        t1_right_gtmask = 255* torch.ones_like(t1_right_depth)
        t1_right_gtmask[t1_right_depth[:,0,y_base,x_base]>t0_right_depth[:,0,y_idx, x_idx]] =0
        save_path = rgb_right_path[1].split('/')
        save_path[-3] = "forward_temporal_mask"
        #print("t1_right_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path), t1_right_gtmask.squeeze().cpu().numpy())
       




        # backward_temporal mask
        ### left view
        cam_points = self.layer_backproject_depth(t0_left_depth, left_invK44)
        #pix_coordi = self.layer_project(cam_points,  left_K44, cTr) 
        #I_wrL = F.grid_sample(t1_left_img, pix_coordi, mode='bilinear', padding_mode="border")
        #t0_left_gtmask = torch.zeros((b,c,h,w))
        #t0_left_gtmask[torch.abs(I_wrL-t0_left_img)<5] = 255
        #t0_left_gtmask = t0_left_gtmask.mean(-3,False)
        P = torch.matmul(left_K44, cTr)[:, :3, :]
        cam_points = torch.matmul(P, cam_points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
        pix_coords = pix_coords.view(b, 2, h, w)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coordi = torch.round(pix_coords)
        x_idx = torch.clamp(pix_coordi[..., 0], 0, w-1).long()
        y_idx =torch.clamp( pix_coordi[..., 1], 0, h-1).long()
        t0_left_gtmask = 255* torch.ones_like(t0_left_depth)
        t0_left_gtmask[t0_left_depth[:,0,y_base,x_base]<t1_left_depth[:,0,y_idx, x_idx]] =0
        save_path = rgb_left_path[0].split('/')
        save_path[-3] = "backward_temporal_mask"
        #print("t0_left_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path), t0_left_gtmask.squeeze().cpu().numpy())

        ### on right view
        cam_points = self.layer_backproject_depth(t0_right_depth, right_invK44)
        #pix_coordi = self.layer_project(cam_points,  right_K44, cTr_right) 
        #I_wrL = F.grid_sample(t1_right_img, pix_coordi, mode='bilinear', padding_mode="border")
        #t0_right_gtmask = torch.zeros((b,c,h,w))
        #t0_right_gtmask[torch.abs(I_wrL-t0_left_img)<5] = 255
        #t0_right_gtmask = t0_right_gtmask.mean(-3,False)
        P = torch.matmul(right_K44, cTr_right)[:, :3, :]
        cam_points = torch.matmul(P, cam_points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
        pix_coords = pix_coords.view(b, 2, h, w)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coordi = torch.round(pix_coords)
        x_idx = torch.clamp(pix_coordi[..., 0], 0, w-1).long()
        y_idx =torch.clamp( pix_coordi[..., 1], 0, h-1).long()
        t0_right_gtmask = 255* torch.ones_like(t0_right_depth)
        t0_right_gtmask[t0_right_depth[:,0,y_base,x_base]<t1_right_depth[:,0,y_idx, x_idx]] =0
        save_path = rgb_right_path[0].split('/')
        save_path[-3] = "backward_temporal_mask"
        #print("t0_right_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path), t0_right_gtmask.squeeze().cpu().numpy(), )


        # stereo occlusion mask

        ### for time 0  reference view 
        #print("focal",focal)
        #print("baseline",baseline)
        right_disp = focal*baseline / t0_right_depth
        #print(right_disp)
        #t0_reconstructed_right, t0_mask_right = self.generate_image_right(t0_left_img, t0_right_img, right_disp)
        t0_mask_right = self.generate_depth_right( t0_right_depth, t0_left_depth, right_disp)
        save_path = rgb_right_path[0].split('/')
        save_path[-3] = "stereo_occlusion_mask"
        #print("t0_right_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path), t0_mask_right.squeeze().cpu().numpy(), )

        left_disp = focal*baseline / t0_left_depth
        #t0_reconstructed_left, t0_mask_left = self.generate_image_left(t0_right_img, t0_left_img, left_disp)
        t0_mask_left = self.generate_depth_left(t0_left_depth, t0_right_depth, left_disp)
        save_path = rgb_left_path[0].split('/')
        save_path[-3] = "stereo_occlusion_mask"
        #print("t0_left_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path),t0_mask_left.squeeze().cpu().numpy())

        ### for time 1  current view 
        right_disp = focal*baseline / t1_right_depth
        #t1_reconstructed_right, t1_mask_right = self.generate_image_right(t1_left_img, t1_right_img, right_disp)
        t1_mask_right = self.generate_depth_right( t1_right_depth,t1_left_depth, right_disp)
        save_path = rgb_right_path[1].split('/')
        save_path[-3] = "stereo_occlusion_mask"
        #print("t1_right_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path), t1_mask_right.squeeze().cpu().numpy(), )

        left_disp = focal*baseline / t1_left_depth
        #t1_reconstructed_left, t1_mask_left = self.generate_image_left(t1_right_img, t1_left_img, left_disp)
        t1_mask_left = self.generate_depth_left( t1_left_depth, t1_right_depth, left_disp)
        save_path = rgb_left_path[1].split('/')
        save_path[-3] = "stereo_occlusion_mask"
        #print("t1_left_gtmask saved in: ", '/'.join(save_path))
        if not os.path.exists('/'.join(save_path[:-1])):
            os.makedirs('/'.join(save_path[:-1]))
        cv2.imwrite('/'.join(save_path), t1_mask_left.squeeze().cpu().numpy(), )

    def generate_image_left(self, source_img, target_img, disp,mod="dense"):
        return self.apply_disparity(source_img,target_img, -disp,mod, name="left_mask") # from left view to right view projection, x_base-left_disparity
    
    def generate_image_right(self, source_img, target_img, disp,mod="dense"):
        return self.apply_disparity(source_img, target_img,  disp,mod, name="right_mask") 
    def apply_disparity(self, img, target_img, disp,mod="dense",name=None):
        batch_size, _, height, width = img.size()
        y_base, x_base = torch.meshgrid(
                                torch.linspace(0.0 , height - 1.0 , height),
                                torch.linspace(0.0,   width - 1.0,  width))
        y_base, x_base =  y_base.unsqueeze(0).repeat(batch_size,1,1).type_as(img), \
                         x_base.unsqueeze(0).repeat(batch_size,1,1).type_as(img)

        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        #x_shifts = x_shifts / 191
        #print("x_shift\n {}".format(x_shifts))
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        #print("flow_field\n {}".format(flow_field))
        # norm for grid_sample 
        flow_field[...,0] /=(width-1)
        flow_field[...,1] /=(height-1)
        flow_field = (flow_field-0.5)*2.0

        output = F.grid_sample(img, flow_field, mode='bilinear',padding_mode='border',align_corners=False)
            #mode='nearest', padding_mode='border',)
        #warp_image_with_zero = F.grid_sample(img, flow_field, mode='bilinear',padding_mode="zeros",align_corners=False)
        visible_mask = torch.zeros_like(img).to(img.device)
        visible_mask[torch.abs(output-target_img)<3] = 255
        visible_mask = torch.mean(visible_mask,-3,False)
  
        return output,visible_mask

    def generate_depth_left(self, source_depth, target_depth, source_disp ):
        return self.apply_depth_disparity(source_depth,target_depth, -source_disp, ) # from left view to right view projection, x_base-left_disparity
    
    def generate_depth_right(self, source_depth, target_depth, source_disp,):
        return self.apply_depth_disparity(source_depth, target_depth,  source_disp, ) 
    def apply_depth_disparity(self, source_depth, target_depth, source_disp, ):
        batch_size, _, height, width = source_depth.size()
        y_base, x_base = torch.meshgrid(
                                torch.linspace(0.0 , height - 1.0 , height),
                                torch.linspace(0.0,   width - 1.0,  width))
        y_base, x_base =  y_base.unsqueeze(0).repeat(batch_size,1,1).long(), \
                         x_base.unsqueeze(0).repeat(batch_size,1,1).long()

        x_shifts = source_disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel

        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        pix_coordi = torch.round(flow_field)
        x_idx = torch.clamp(pix_coordi[..., 0], 0, width-1).long()
        y_idx =torch.clamp( pix_coordi[..., 1], 0, height-1).long()
        
        occlu_mask = 255*torch.ones_like(source_depth)
        occlu_mask[source_depth[:,0,y_base, x_base]> target_depth[:,0,y_idx,x_idx]] = 0 

 
  
        return occlu_mask