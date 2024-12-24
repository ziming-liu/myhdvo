import os.path as osp
from typing import Sequence
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from .evaluation_utils import *
import os
import glob
from sys import prefix
import torch
import pandas as pd
import  time
import numpy as np
import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from hdvo.utils import register_module_hooks, get_root_logger
from torch.utils.data import Dataset
from .pipelines import Compose

#from ..core import (mean_average_precision, mean_class_accuracy,rmse,
#                    rel,relsqr, log10, rmsedepth, rmsedepthlog, correct,
#                    mmit_mean_average_precision, top_k_accuracy)

from ..core.geometry.camera_modules import Intrinsics

from .base import BaseDataset
from .registry import DATASETS
#from ..models.geometry import *

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
class KittiDepthStereoDataset(Dataset):
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

    def __init__(self,  pipeline, 
                 seq_len_range, 
                 ann_file=None,
                 overlap=1, 
                 data_prefix=None,
                 frame_shifts=[[0,],],
                 depth_range=(1,80),
                 end_id=-1,
                 crop_test_image="garg",
                 scale=(848,256),
                 stereo_id=["Image_02","Image_03"], 
                 test_mode=False,
                 filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png', 
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 modality='RGB',
                 sample_by_class=False,
                 power=None,
                 regression=True,
                 sample_times=1, 
                 pad_y=False, 
                  **kwargs):

        self.end_id = end_id
        self.test_mode = test_mode
        self.data_prefix = data_prefix
        self.frame_shifts = frame_shifts
        self.depth_range = depth_range
        self.crop_test_image = crop_test_image
        self.pose_dirs = []
        self.cam_intrin_dirs = []
        self.seq_raw_dirs = []
        self.seq_depth_dirs = []
        self.raw_data_path = os.path.join(data_prefix, "kitti_raw_data")
        
        #self.pose_dir = osp.join(data_prefix, 'pose_GT')
        #self.seq_dir = osp.join(data_prefix, 'sequences')
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
        
        #self.seq_len_list = list(self.video_infos.seq_len)
        #self.image_arr = np.asarray(self.video_infos.frame_paths)  # image paths
        #self.groundtruth_arr = np.asarray(self.video_infos.pose)
        self.regression = regression
        #self.start_index = start_index
        self.filename_tmpl = filename_tmpl
        self.depth_filename_tmpl = depth_filename_tmpl
        self.with_offset = with_offset
        self.origin_size = [] # include N sequences's image/depth size
        self.origin_h_w = None
        self.video_infos = self.load_annotations()

        self.ann_file = ann_file
        self.data_prefix = data_prefix# osp.realpath(
            #data_prefix) if data_prefix is not None and osp.isdir(
            #    data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        #self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        #self.video_infos = self.load_annotations()
        #print(" len of video infos -> {}".format(len(self.video_infos)))
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()
        super().__init__()

        
    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)


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
        #results['depth_filename_tmpl'] = self.depth_filename_tmpl
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
        results['modality'] = self.modality
        #results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl
        #results['depth_filename_tmpl'] = self.depth_filename_tmpl
        #results['pose'] = self._process_gt_pose(results['pose']) # test mode don't need gt pose for kitti depth dataset
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
        # the T_tar == 0Tc   T_ref == 0Tr
        relatvie_cTr = torch.mm(torch.linalg.inv(T_tar.float()), T_ref.float()).float() # cTr
        relatvie_rTc = torch.mm(torch.linalg.inv(T_ref.float()), T_tar.float()).float() # rTc
        #print("relative pose")
        #print(relatvie_pose)
        #exit()
        # print('Item after transform: ' + str(index) + '   ' + str(groundtruth_sequence))
        relative_pose = torch.stack([relatvie_cTr,relatvie_rTc])
        return relative_pose

    def load_kitti_depth_odom_intrinsics(self, camera_intrinsic_path):
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
            left_cam_intri = raw_intrisic[25]
            right_cam_intri = raw_intrisic[33]
            assert left_cam_intri[:9] == "P_rect_02"
            assert right_cam_intri[:9] == "P_rect_03"

            #for i, item in enumerate(left_cam_intri):
            # left 
            left_line_split = [float(value) for value in left_cam_intri.strip().split(' ')[-12:] ]
            assert len(left_line_split) == 12
            left_cam_intrinsics = np.array(left_line_split).reshape((3,4))
            left_K = np.eye(4)
            left_K[:3,:4] = left_cam_intrinsics
            #left_invK = np.linalg.inv(left_K)
            
            # right
            right_line_split = [float(value) for value in right_cam_intri.strip().split(' ')[-12:] ]
            assert len(right_line_split) == 12
            right_cam_intrinsics = np.array(right_line_split).reshape((3,4))
            right_K = np.eye(4)
            right_K[:3,:4] = right_cam_intrinsics
            #right_invK = np.linalg.inv(right_K)
        
        output_cam_intrinsics = np.stack( (left_K[:3,:3], right_K[:3,:3])  )
        return output_cam_intrinsics

    def get_focal_length_baseline(self, calib_dir, cam):
        with open(calib_dir, 'r') as f:
            cam2cam = f.readlines()
        P2_rect = cam2cam[26-1]
        assert P2_rect[:9] == "P_rect_02"
        P2_rect = np.array([float(P2_rect.strip().split()[-12:][i]) for i in range(12)  ])
        P2_rect = P2_rect.reshape(3,4)

        P3_rect = cam2cam[34-1]
        assert P3_rect[:9] == "P_rect_03"
        P3_rect = np.array([float(P3_rect.strip().split()[-12:][i]) for i in range(12)  ])
        P3_rect = P3_rect.reshape(3,4)

        # cam 2 is left of camera 0  -6cm
        # cam 3 is to the right  +54cm
        b2 = P2_rect[0,3] / -P2_rect[0,0]
        b3 = P3_rect[0,3] / -P3_rect[0,0]
        baseline = b3-b2

        if cam==2:
            focal_length = P2_rect[0,0]
        elif cam==3:
            focal_length = P3_rect[0,0]

        return focal_length.astype(np.float64), baseline.astype(np.float64)
    def load_annotations(self,):
        print("ann file: {}".format(self.ann_file))
        if self.ann_file is None:
            raise ValueError
        else:
            print("using annotation files {} ", self.ann_file, "......")
            X_rgb_path_left, X_rgb_path_right, Y_pose_abs = [], [], []
            X_depth_path_left, X_depth_path_right = [], []
            X_len = []
            X_dir = []
            X_depth_dir = []
            X_intrinsics = []
            X_start_ind = []
            X_focals = []
            X_baselines = []
            X_total_frames = []
            video_infos = []

            left_seq_dirs = dict()
            right_seq_dirs = dict()
            self.folder_list = []
            num_samples = 0 

            if self.ann_file.split('/')[-2] in ["eigen_monodepth", "eigen_with_gt"]:
                
                with open(self.ann_file, 'r') as f:
                    lines = f.readlines()
                    num_samples = len(lines)
                    for i, line in enumerate(lines):
                        left_rgb, right_rgb =  line.strip().split()
                        left_rgb = left_rgb.split(".")[0]+".png"
                        right_rgb = right_rgb.split(".")[0]+".png"
                        assert "image_02" in left_rgb and "image_03" not in left_rgb
                        assert "image_03" in right_rgb and  "image_02" not in right_rgb
                        seq_id = left_rgb.split('/')[1]
                        if seq_id not in left_seq_dirs:
                            self.folder_list.append(seq_id)
                            left_seq_dirs[seq_id] = []
                            right_seq_dirs[seq_id] = []
                        left_seq_dirs[seq_id].append(os.path.join(self.data_prefix, "kitti_raw_data",left_rgb))
                        right_seq_dirs[seq_id].append(os.path.join(self.data_prefix, "kitti_raw_data", right_rgb))
                    

            elif self.ann_file.split('/')[-2]  in ["eigen_zhou", "eigen_full", "eigen_benchmark", ]:
                 
                with open(self.ann_file, 'r') as f:
                    lines = f.readlines()
                    num_samples = int(len(lines) / 2 )
                    for i, line in enumerate(lines):
                        pre_path, img_id, stereo_side = line.strip().split()
                        if stereo_side == 'r':
                            continue
                        assert stereo_side =='l', "remove the repeated annotations"
        
                        left_rgb = os.path.join(pre_path,"image_02/data", self.filename_tmpl.format(int(img_id)) )
                        right_rgb = os.path.join(pre_path,"image_03/data", self.filename_tmpl.format(int(img_id)) )

                        seq_id = pre_path.split('/')[1]
                        if seq_id not in left_seq_dirs:
                            self.folder_list.append(seq_id)
                            left_seq_dirs[seq_id] = []
                            right_seq_dirs[seq_id] = []
                        left_seq_dirs[seq_id].append(os.path.join(self.data_prefix, "kitti_raw_data",left_rgb))
                        right_seq_dirs[seq_id].append(os.path.join(self.data_prefix, "kitti_raw_data", right_rgb))
            else:
                print(" annotation file {} is wrong".format(self.ann_file))
                raise ValueError

            gt_pose_raw_idx = [ 
            "00: 2011_10_03_drive_0027 000000 004540",
            "01: 2011_10_03_drive_0042 000000 001100",
            "02: 2011_10_03_drive_0034 000000 004660",
            "03: 2011_09_26_drive_0067 000000 000800",
            "04: 2011_09_30_drive_0016 000000 000270",
            "05: 2011_09_30_drive_0018 000000 002760",
            "06: 2011_09_30_drive_0020 000000 001100",
            "07: 2011_09_30_drive_0027 000000 001100",
            "08: 2011_09_30_drive_0028 001100 005170",
            "09: 2011_09_30_drive_0033 000000 001590",
            "10: 2011_09_30_drive_0034 000000 001200",
            ]
            gt_pose_idx = dict()
            for gtposeidx in range(len(gt_pose_raw_idx)):
                odometry_id, raw_id, start_frameid, end_frameid = gt_pose_raw_idx[gtposeidx].split(' ')
                for frame_k in range(int(start_frameid), int(end_frameid)+1):
                    gt_pose_idx[raw_id+"_sync_"+"{:0>6}".format(str(frame_k))] = odometry_id.split(':')[0]
            #print("gt pose idx >>> \n ", gt_pose_idx)
            # load all gt pose one time. 
            gt_pose_values = dict()
            for odometry_idx in range(11):
                pose_path = "/data/acentauri/user/ziliu/data/kitti_odometry/pose_GT" + "/{:0>2}.txt".format(odometry_idx)
                # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
                with open(pose_path, 'r') as pf:
                    raw_poses = pf.readlines()
                    left_poses_abs = [np.array([ float(value) for value in pose_item.strip().split(' ')]).reshape((3,4)) for pose_i, pose_item in enumerate(raw_poses)]
                    left_poses_abs = [ np.concatenate((pose,np.array([0,0,0,1]).reshape(1,4) ),0) for pose in left_poses_abs]
                    #num_frames = len(left_poses_abs)
                    print(pose_path, ">>")
                    print("num poses: {} ".format(len(left_poses_abs)))
                    gt_pose_values["{:0>2}".format(odometry_idx)] = left_poses_abs
            


            X_rgb_path_left, X_rgb_path_right, Y_pose_abs = [], [], []
            Y_stereo_poses = []
            X_depth_path_left, X_depth_path_right = [], []
            X_len = []
            X_dir = []
            X_depth_dir = []
            X_intrinsics = []
            X_start_ind = []
            X_total_frames = []
            video_infos = []
            X_focals = []
            X_baselines = []
            print( " total {} # seqs {} # samples will be loaded".format(len(self.folder_list),num_samples))
            total_gt_pose_samples = 0
            for seq_i, seq_id in enumerate(self.folder_list):
                start_t = time.time()
                #num_frames = -1
                calib_path = os.path.join(self.data_prefix, "kitti_raw_data", '_'.join(seq_id.split('_')[:3]) , "calib_cam_to_cam.txt")
                cam_intrinsics = self.load_kitti_depth_odom_intrinsics(calib_path) # a dict of camera 0, 1, 2, 3, tr 
                # TODO: keep both left and right intrinsics
                # focal and baseline of left camera
                focal, baseline = self.get_focal_length_baseline(calib_path,2)
                
                # sort image paths 
                left_seq_dirs[seq_id] = left_seq_dirs[seq_id]
                right_seq_dirs[seq_id] =  right_seq_dirs[seq_id]
                #num_frames = len(left_seq_dirs[seq_id])
                img_ids = [ int(pp.split("/")[-1].split(".")[0])  for pp in left_seq_dirs[seq_id]]
                
                
                num_frames = len(os.listdir( os.path.join(os.path.join(self.data_prefix, "kitti_raw_data", \
                                    '_'.join(seq_id.split('_')[:3]), seq_id, "image_02/data"))))
                for img_idx, img_id in enumerate(img_ids):
                    
                    # all shifts is transofrmed to [[],[],[],,,] each group size is 2
                    for shift_group_k in range(len(self.frame_shifts)):
                        shift_group = self.frame_shifts[shift_group_k]
                        if self.test_mode:
                            assert len(shift_group) == 1 and shift_group[0]==0
                        move_forward = 0 # if excess the end frame, we should move forward entirtely, 
                        if img_id + shift_group[-1] > num_frames-1: # last frame idx > num frames -1
                            move_forward = img_id + shift_group[-1] - num_frames +1 
                        #print("img_id + shift_group[-1]>> ", img_id, "+", shift_group[-1],"num_frames: ", num_frames, " move forward >> ", move_forward)

                        move_backward = 0
                        if img_id + shift_group[0] < 0: # first frame idx <0, 
                            move_backward = 0 - (img_id + shift_group[0])
                        #print("img_id + shift_group[0]>> ", img_id, "+", shift_group[0], " move back >> ", move_backward)
                        flag_have_gt_pose = 1
                        # for each samples, load "shifts number" frames
                        one_left_rgb= []
                        one_right_rgb = []
                        one_left_depth = []
                        one_right_depth = []
                        one_gt_pose = []
                        #print("shift group >> ", shift_group)
                        # generate one sample
                        for shift in shift_group:
                            sample_img_id = img_id + shift - move_forward + move_backward
                            #print("img id >> ", img_id)
                            #print("sample_img id >> ", sample_img_id)
                            sample_left_rgb_path = os.path.join(os.path.join(self.data_prefix, "kitti_raw_data",
                                    '_'.join(seq_id.split('_')[:3]), seq_id, "image_02/data", self.filename_tmpl.format(sample_img_id)))
                            sample_right_rgb_path = os.path.join(os.path.join(self.data_prefix, "kitti_raw_data",
                                    '_'.join(seq_id.split('_')[:3]), seq_id, "image_03/data", self.filename_tmpl.format(sample_img_id)))
                            sample_left_depth_path = os.path.join(self.data_prefix,"kitti_raw_depth",seq_id,
                                                        "proj_depth/groundtruth/image_02", self.depth_filename_tmpl.format(sample_img_id))
                            sample_right_depth_path = os.path.join(self.data_prefix,"kitti_raw_depth",seq_id,
                                                            "proj_depth/groundtruth/image_03", self.depth_filename_tmpl.format(sample_img_id))
                            
                            # add pose
                            if seq_id+"_"+"{:0>6}".format(sample_img_id) not in gt_pose_idx.keys():
                                #print("there is not GT pose for ", seq_id+"_"+"{:0>6}".format(sample_img_id) )
                                flag_have_gt_pose = 0
                                one_gt_pose.append(np.eye(4))
                            else:
                                odom_seq = gt_pose_idx[seq_id+"_"+"{:0>6}".format(sample_img_id)]
                                if int(odom_seq) ==8:
                                    odom_seq_frameid  = int(sample_img_id) - 1100
                                else:
                                    odom_seq_frameid  = int(sample_img_id)
                                one_gt_pose.append(gt_pose_values[odom_seq][odom_seq_frameid] )
                                #print(" raw image >> ",sample_left_rgb_path , " in  odometry sqe, id >> ",odom_seq, odom_seq_frameid, " load pose >> ", gt_pose_values[odom_seq][odom_seq_frameid] )
                            
                            if not os.path.exists(sample_left_rgb_path) and not  os.path.exists(sample_left_depth_path):
                                # if there is no this sample in dataset, don't add it to annotations.
                                print("sample_left_rgb_path>> ",sample_left_rgb_path, "  sample_left_depth_path>> ", sample_left_depth_path )
                                raise ValueError
                                
                            one_right_rgb.append(sample_right_rgb_path)
                            one_left_rgb.append(sample_left_rgb_path)
                            one_left_depth.append(sample_left_depth_path)
                            one_right_depth.append(sample_right_depth_path)
                        
                        # append this one sample into all_annotations
                        
                        if not self.test_mode:
                            if flag_have_gt_pose == 0: # not gt pose, use identity pose of a same frame
                                break # ingnore all sampels of depth kitti, which can't find correspoding  GT Pose in odometry annoations.
                                #one_right_rgb = [one_right_rgb[0] for  _ in range(len(sshift_group))]
                                #one_left_rgb = [one_left_rgb[0] for  _ in range(len(shift_group))]
                                #one_left_depth = [one_left_depth[0] for  _ in range(len(shift_group))]
                                #one_right_depth = [one_right_depth[0] for  _ in range(len(sshift_group))]
                            else:
                                total_gt_pose_samples +=1
                        #print("one left rgb >> ",one_left_rgb)
                        X_rgb_path_left.append(one_left_rgb)
                        X_rgb_path_right.append(one_right_rgb)
                        X_depth_path_left.append(one_left_depth)
                        X_depth_path_right.append(one_right_depth)
                        if not self.test_mode:
                            Y_pose_abs.append(one_gt_pose)
                        X_start_ind.append(int(sample_img_id))
                        X_total_frames.append(num_frames)
                        if self.origin_h_w is None:
                            # read heigh and width
                            import cv2
                            example_sam = cv2.imread(X_rgb_path_left[-1][-1])
                            H,W,C = example_sam.shape
                            self.origin_h_w = (H,W)

                        #if len(self.frame_shifts) ==1:
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

                        Y_stereo_poses.append(stereo_pose)
                        X_len.append(len(shift_group))
                        X_dir.append(os.path.join(self.data_prefix, "kitti_raw_data",
                                    '_'.join(seq_id.split('_')[:3]), seq_id)) 
                        X_depth_dir.append(os.path.join(self.data_prefix,"kitti_raw_depth",seq_id))
                        #X_start_ind.append(int(f_rgb_paths_left[i].split('/')[-1].split('.')[0]))
                        #X_total_frames.append(len(self.frame_shifts))
                        X_intrinsics.append(cam_intrinsics)
                        X_focals.append(focal)
                        X_baselines.append(baseline)
                        self.origin_size.append(self.origin_h_w)

            
            #vis_eigen_depth = "/home/ziliu/vis_eigen_depth"
            ##vis_eigen_img = "/home/ziliu/vis_eigen_img"
            #if not os.path.exists(vis_eigen_depth):
            #    os.makedirs(vis_eigen_depth)
            #    os.makedirs(vis_eigen_img)
            print("there is total_gt_pose_samples >>>>>> ", total_gt_pose_samples)
            print("load total {} samples".format(len(X_rgb_path_left)))
            #print(X_rgb_path_left)
            for i in range(len(X_rgb_path_left)):
                #if not os.path.isfile(X_depth_path_left[i]):
                    # filter the samples that don;t have official GT depth  652 / 697
                #    continue
                #from shutil import copyfile
                #print(X_rgb_path_left[i][0])
                #copyfile(X_rgb_path_left[i][0], os.path.join(vis_eigen_img, '{:0>10}.png'.format(i) ) )
                #copyfile(X_depth_path_left[i][0], os.path.join(vis_eigen_depth, '{:0>10}.png'.format(i) ))
                if self.test_mode:
                    video_infos.append(dict(total_frames=X_total_frames[i],
                                        seq_len=X_len[i],
                                        original_size = self.origin_size[i],
                                        frame_dir=X_dir[i],
                                        depth_dir=X_depth_dir[i],
                                        start_index=X_start_ind[i],
                                        intrinsics = X_intrinsics[i],
                                        focal=X_focals[i],
                                        baseline=X_baselines[i],
                                        stereo_pose=Y_stereo_poses[i],
                                        #start_index=0,
                                        frame_paths=[X_rgb_path_left[i], X_rgb_path_right[i]] if len(self.stereo_id)==2 \
                                        else [X_rgb_path_left[i], ],
                                        left_frame_paths=X_rgb_path_left[i],
                                        right_frame_paths = X_rgb_path_right[i],
                                        #left_depth_paths=X_depth_path_left[i],
                                        #right_depth_paths= X_depth_path_right[i],
                                            ))
                else:
                    assert len(X_rgb_path_left[i]) == 2
                    pose=[Y_pose_abs[i][0], Y_pose_abs[i][-1]]
                    right_pose = [np.matmul(Y_stereo_poses[i][0], np.matmul(Y_pose_abs[i][0], Y_stereo_poses[i][1])), np.matmul(Y_stereo_poses[i][0], np.matmul(Y_pose_abs[i][-1], Y_stereo_poses[i][1])) ] # oTLr \cdot LTR;  oTLc \cdot LTR
                    video_infos.append(dict(total_frames=X_total_frames[i],
                                            seq_len=X_len[i],
                                            original_size = self.origin_size[i],
                                            frame_dir=X_dir[i],
                                            depth_dir=X_depth_dir[i],
                                            start_index=X_start_ind[i],
                                            intrinsics = X_intrinsics[i],
                                            focal=X_focals[i],
                                            baseline=X_baselines[i],
                                            stereo_pose=Y_stereo_poses[i],
                                            pose = pose,
                                            right_pose = right_pose,
                                            #start_index=0,
                                            frame_paths=[X_rgb_path_left[i], X_rgb_path_right[i]] if len(self.stereo_id)==2 \
                                            else [X_rgb_path_left[i], ],
                                            left_frame_paths=X_rgb_path_left[i],
                                            right_frame_paths = X_rgb_path_right[i],
                                            #left_depth_paths=X_depth_path_left[i],
                                            #right_depth_paths= X_depth_path_right[i],
                                                ))

        if self.test_mode:
            return video_infos[:self.end_id] #[:5]#[:100]
        return video_infos[:self.end_id] #[:int(0.5*len(video_infos))]

    def evaluate(self,
                 results,
                 eval_tasks=['depth'],
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
        # evaluate KITTI Depth: two modes- kitti official or eigen
        ##########################################
        #if "depth" in eval_tasks:
            
        split =  'eigen'
        if split=='kitti':
            gt_path = '/data/acentauri/user/ziliu/data/stereo_matching_data/StereoMatching/KITTI-2015/'
        elif split =='eigen':
            gt_path = '/data/acentauri/user/ziliu/data/kitti_raw_data/'
        else:
            raise ValueError
        print("### start to evaluate KITTI DEPTH, {} split ###".format(split))
        min_depth = self.depth_range[0] #2.018
        max_depth = self.depth_range[1]
        if self.crop_test_image is None:
            garg_crop, eigen_crop = False, False
        elif  self.crop_test_image == "eigen":
            garg_crop, eigen_crop = False, True
        elif self.crop_test_image == "garg":
            garg_crop, eigen_crop = True, False
        print(f">> test image with {self.crop_test_image} crop")
        V = results[0][0].shape[0]
        for view_idx in range(V):
            print("### FOR the view ", view_idx)
            pred_disparities = [results[0][bi][view_idx,:,:] for bi in range(len(results[0]))] # shape: batchsize, h, w # keep view 0 dimension 
            #frame_dir = results[-1]
        
            if split == 'kitti':
                num_samples = 200
                
                gt_disparities = load_gt_disp_kitti(gt_path)
                gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

            elif split == 'eigen':
                num_samples = 697
                #test_files = read_text_lines('/data/acentauri/user/ziliu/data/depth_splits/eigen_test_files.txt')
                test_files = read_text_lines(self.ann_file)
                gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path)

                num_test = len(im_files)
                gt_depths = []
                pred_depths = []
                print("# load gt depth")
                if len(pred_disparities) < num_samples:
                    num_samples = len(pred_disparities)
                for t_id in tqdm(range(num_samples)):
                    camera_id = cams[t_id]  # 2 is left, 3 is right
                    #print("gt calib {}".format(gt_calib[t_id]))
                    depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
                    gt_depths.append(depth.astype(np.float32))
                    # im_sizes: (h, w)
                    disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
                    #print("origin disp \n {}".format(disp_pred))
                    #TODO: why x width, scale disparity for monodepth?? acfnet don;t need this. 
                    #disp_pred = disp_pred * disp_pred.shape[1]
                    #print("x width disp \n {}".format(disp_pred))
                    # need to convert from disparity to depth
                    #focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
                    #depth_pred = (baseline * focal_length) / disp_pred
                    # cm to m 
                    depth_pred = disp_pred # for hybrid vo return, the return disp_pred is already depth.
                    depth_pred[np.isinf(depth_pred)] = 0
                    #print("depth pred  \n {}".format(depth_pred))
                    pred_depths.append(depth_pred)
                    
                    #assert frame_dir[t_id] == '/'.join(im_files[t_id].split('/')[:-3]), "{}".format('/'.join(im_files[t_id].split('/')[:-3]))
            pred_depths = pred_depths[:num_samples]
            assert len(pred_depths) == len(gt_depths)
            rms     = np.zeros(num_samples, np.float32)
            log_rms = np.zeros(num_samples, np.float32)
            abs_rel = np.zeros(num_samples, np.float32)
            sq_rel  = np.zeros(num_samples, np.float32)
            d1_all  = np.zeros(num_samples, np.float32)
            a1      = np.zeros(num_samples, np.float32)
            a2      = np.zeros(num_samples, np.float32)
            a3      = np.zeros(num_samples, np.float32)
            print("# compute metrics...")
            for i in tqdm(range(num_samples)):
                
                gt_depth = gt_depths[i]
                pred_depth = pred_depths[i]
                pred_depth[pred_depth < min_depth] = min_depth
                pred_depth[pred_depth > max_depth] = max_depth

                if split == 'eigen':
                    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

                    
                    if garg_crop or eigen_crop:
                        gt_height, gt_width = gt_depth.shape

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

                if split == 'kitti':
                    gt_disp = gt_disparities[i]
                    mask = gt_disp > 0
                    pred_disp = pred_disparities_resized[i]

                    disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
                    bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
                    d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

                abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
            
            msg = f'Evaluating  view {view_idx}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
            print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean()*100, a2.mean()*100, a3.mean()*100))
            eval_res = {'abs_rel':abs_rel.mean(), 'sq_rel': sq_rel.mean(), 'rms':rms.mean(), 'log_rms': log_rms.mean(), 'd1_all':d1_all.mean(), 'a1':a1.mean()*100, 'a2':a2.mean()*100, 'a3': a3.mean()*100}
            log_msg = []
            log_msg.append(f'\n evaluation results')
            for k, acc in eval_res.items():
                #depth_eval_results[f'_{k}'] = acc
                log_msg.append(f'  {k}: {acc:.4f}  ')
                self.logger.info(f'  {k}: {acc:.4f}  ')
            log_msg = ' '.join(log_msg)
            print_log(log_msg, logger=logger)
        
        pose_eval_results = OrderedDict()
        
            



        return tuple([depth_eval_results,pose_eval_results])


