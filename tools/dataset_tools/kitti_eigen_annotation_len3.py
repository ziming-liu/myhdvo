import numpy as np
import cv2
import json
import os 
from tqdm import tqdm
import time
# /home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_train_files.txt
# /home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt
# /home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_val_files.txt

def get_pose():
    gt_pose_raw_idx = [ 
    "00: 2011_10_03_drive_0027 000000 004540",
    "01: 2011_10_03_drive_0042 000000 001100",
    "02: 2011_10_03_drive_0034 000000 004660",
    "03: 2011_09_26_drive_0067 000000 000800",
    "04: 2011_09_30_drive_0016 000000 000270",
    "05: 2011_09_30_drive_0018 000000 002760",
    "06: 2011_09_30_drive_0020 000000 001100",
    "07: 2011_09_30_drive_0027 000000 001100",
    "08: 2011_09_30_drive_0028 001100 005170", # need to  - 1100
    "09: 2011_09_30_drive_0033 000000 001590",
    "10: 2011_09_30_drive_0034 000000 001200",
    ]
    raw_imgid_to_odom_imgid = dict()
    for gtposeidx in range(len(gt_pose_raw_idx)):
        odometry_id, raw_id, start_frameid, end_frameid = gt_pose_raw_idx[gtposeidx].split(' ')
        odometry_id = odometry_id.split(':')[0] # str '00'
        raw_id = raw_id+"_sync" # str '2011_10_03_drive_0027_sync'
        odom_imgid = 0
        for imgid in range(int(start_frameid), int(end_frameid)):
            raw_imgid_to_odom_imgid[raw_id[:10] +'/' + raw_id + "/image_02/data"+"/{:0>10}.png".format(imgid)] = "{:0>2}".format(odometry_id) + "/{:0>6}.png".format(odom_imgid)
            odom_imgid += 1
        

        #odometry_dir = "/home/ziliu/mydata/kitti_odometry/sequences/" + odometry_id + "/image_2"
        #raw_img_dir = "/home/ziliu/mydata/kitti_raw_data/" +raw_id[:10] +'/' + raw_id+"_sync" + "/image_02/data"

        #assert len(os.listdir(odometry_dir)) == len(os.listdir(raw_img_dir)), f"ERROR in seq {odometry_id} >\n odometry_dir vs  raw_img_dir: {len(os.listdir(odometry_dir))} vs {len(os.listdir(raw_img_dir))} \n path {odometry_dir} \n path {raw_img_dir}"
        # we have to align the timestamp of odometry and raw data
        # /home/ziliu/mydata/kitti_odometry/sequences/00/times.txt
        # /home/ziliu/mydata/kitti_raw_data/2011_10_03/2011_10_03_drive_0027_sync/image_02/timestamps.txt
        
   # load all gt abs pose from KITTI odometry GT pose.txt
    gtpose_ann = [] #dict() # {odometry_id: [4,4]}
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
            gtpose_ann.append(left_poses_abs)
            #gtpose_ann["{:0>2}".format(odometry_idx)] = left_poses_abs
    return gtpose_ann, raw_imgid_to_odom_imgid



def load_kitti_depth_odom_intrinsics( camera_intrinsic_path, gray=False):
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
        if gray:
            left_cam_intri = raw_intrisic[10-1]
            right_cam_intri = raw_intrisic[18-1]
            assert left_cam_intri[:9] == "P_rect_00"
            assert right_cam_intri[:9] == "P_rect_01"
        else:
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

def get_focal_length_baseline(calib_dir, cam):
    with open(calib_dir, 'r') as f:
        cam2cam = f.readlines()

    P0_rect = cam2cam[10-1]
    assert P0_rect[:9] == "P_rect_00"
    P0_rect = np.array([float(P0_rect.strip().split()[-12:][i]) for i in range(12)  ])
    P0_rect = P0_rect.reshape(3,4)

    P1_rect = cam2cam[18-1]
    assert P1_rect[:9] == "P_rect_01"
    P1_rect = np.array([float(P1_rect.strip().split()[-12:][i]) for i in range(12)  ])
    P1_rect = P1_rect.reshape(3,4)

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
    baseline_23 = b3-b2

    #if cam==2:
    focal_length_2 = P2_rect[0,0]
    #elif cam==3:
    focal_length_3 = P3_rect[0,0]

    b0 = P0_rect[0,3] / -P0_rect[0,0]
    b1 = P1_rect[0,3] / -P1_rect[0,0]
    baseline_01 = b1-b0

    focal_length_0 = P0_rect[0,0]
    focal_length_1 = P1_rect[0,0]

    return focal_length_0.astype(np.float32), focal_length_1.astype(np.float32), \
        focal_length_2.astype(np.float32), focal_length_3.astype(np.float32), \
            baseline_01.astype(np.float32), baseline_23.astype(np.float32)

def count_cam01_mean_std(image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_train_files.txt',
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_raw_data', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png', selfsup=True ):
    gtpose_ann, raw_imgid_to_odom_imgid = get_pose()
    
    outputs = []
    
    means = 0
    stds = 0
    with open(image_ann_path, 'r') as f:
        image_anns = f.readlines()
        for img_ann in tqdm(image_anns):
            # e.g. 2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000215.jpg 2011_09_26/2011_09_26_drive_0070_sync/image_03/data/0000000215.jpg
            left_img_path, right_img_path = img_ann.strip().split(' ')
            left_img_path = left_img_path.replace("image_02", "image_00")
            right_img_path = right_img_path.replace("image_03", "image_01")
            left_img_path = left_img_path.replace("jpg", "png")
            right_img_path = right_img_path.replace("jpg", "png")
            assert os.path.exists(os.path.join(data_prefix, left_img_path)), f"{os.path.join(data_prefix, left_img_path)} does not exist"
            left_img = cv2.imread(os.path.join(data_prefix, left_img_path))
            right_img = cv2.imread(os.path.join(data_prefix, right_img_path))
            
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            left_img = cv2.resize(left_img, (1216, 320))
            right_img = cv2.resize(right_img, (1216, 320))
            image_lists = []
            image_lists.append(left_img)
            image_lists.append(right_img)
            image_lists = np.stack(image_lists, axis=0)
            mean = np.mean(image_lists.reshape(-1),)
            std = np.std(image_lists.reshape(-1),)
            means += (mean)
            stds += (std)
    print("mean: ", means/len(image_anns))
    print("std: ", std/len(image_anns))

def count_cam23_mean_std(image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_train_files.txt',
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_raw_data', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png', selfsup=True ):
    gtpose_ann, raw_imgid_to_odom_imgid = get_pose()
    
    outputs = []
    means = []
    stds = []
    with open(image_ann_path, 'r') as f:
        image_anns = f.readlines()
        for img_ann in tqdm(image_anns):
            # e.g. 2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000215.jpg 2011_09_26/2011_09_26_drive_0070_sync/image_03/data/0000000215.jpg
            left_img_path, right_img_path = img_ann.strip().split(' ')
            #left_img_path = left_img_path.replace("image_02", "image_00")
            #right_img_path = right_img_path.replace("image_03", "image_01")
            left_img_path = left_img_path.replace("jpg", "png")
            right_img_path = right_img_path.replace("jpg", "png")
            assert os.path.exists(os.path.join(data_prefix, left_img_path)), f"{os.path.join(data_prefix, left_img_path)} does not exist"
            left_img = cv2.imread(os.path.join(data_prefix, left_img_path))
            right_img = cv2.imread(os.path.join(data_prefix, right_img_path))
            
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            left_img = cv2.resize(left_img, (1216, 320))
            right_img = cv2.resize(right_img, (1216, 320))
            image_lists = []
            image_lists.append(left_img)
            image_lists.append(right_img)
            image_lists = np.stack(image_lists, axis=0)
            mean = np.mean(image_lists.reshape(-1),)
            std = np.std(image_lists.reshape(-1),)
            means += (mean)
            stds += (std)
    print("mean: ", means/len(image_anns))
    print("std: ", std/len(image_anns))


def make_annotations(image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_train_files.txt',
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_raw_data', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png', selfsup=True ):
    gtpose_ann, raw_imgid_to_odom_imgid = get_pose()
    outputs = []
    pre_seq, next_seq, still_seq = 0,0,0
    with open(image_ann_path, 'r') as f:
        image_anns = f.readlines()
        print("raw data len: ", len(image_anns))
        for img_ann in tqdm(image_anns):
            # e.g. 2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000215.jpg 2011_09_26/2011_09_26_drive_0070_sync/image_03/data/0000000215.jpg
            left_img_path, right_img_path = img_ann.strip().split(' ')
            left_img_path, right_img_path = left_img_path.replace("jpg", "png"), right_img_path.replace("jpg", "png")
            date_id = left_img_path.split('/')[0]
            raw_file_id = left_img_path.split('/')[1]
            img_id = left_img_path.split('/')[-1].split('.')[0]
            # K 
            calib_path = os.path.join(data_prefix, date_id , "calib_cam_to_cam.txt")
            cam_intrinsics = load_kitti_depth_odom_intrinsics(calib_path) # a dict of camera 0, 1, 2, 3, tr 
            cam_intrinsics_gray = load_kitti_depth_odom_intrinsics(calib_path, gray=True)
            focal0, focal1, focal2, focal3, baseline01, baseline23 = get_focal_length_baseline(calib_path, 2)
            
            dict_i = dict()
            gt_poses = []
            if left_img_path in raw_imgid_to_odom_imgid:
                # get the next frame and odometry GT pose
                odom_seqimgid = raw_imgid_to_odom_imgid[left_img_path]
                odom_seqid = int(odom_seqimgid.split('/')[-2])
                odom_imgid = int(odom_seqimgid.split('/')[-1].split('.')[0])
                raw_imgid = int(left_img_path.split('/')[-1].split('.')[0])
                cur_id = '{:0>10}.png'.format(raw_imgid)
                # the last frame
                last_id = '{:0>10}.png'.format(raw_imgid-1)
                if os.path.exists(os.path.join(data_prefix, left_img_path.replace(cur_id, last_id))) and odom_imgid-1 >= 0 and odom_imgid-1 < len(gtpose_ann[odom_seqid]):
                    pre_flag = 1
                else:
                    pre_flag = 0
                # the next frame 
                next_id = '{:0>10}.png'.format(raw_imgid+1)
                if os.path.exists(os.path.join(data_prefix, left_img_path.replace(cur_id, next_id))) and odom_imgid+1 >= 0 and odom_imgid+1 < len(gtpose_ann[odom_seqid]):
                    next_flag = 1
                else:
                    next_flag = 0

                if pre_flag == 1 and next_flag == 1:
                    #pre_seq +=1
                    gt_poses = [ gtpose_ann[odom_seqid][odom_imgid-1],
                                            gtpose_ann[odom_seqid][odom_imgid], gtpose_ann[odom_seqid][odom_imgid+1]] 
                    image_2_paths = [os.path.join(data_prefix, left_img_path.replace(cur_id,last_id)),\
                                      os.path.join(data_prefix, left_img_path),\
                                        os.path.join(data_prefix, left_img_path.replace(cur_id,next_id))]
                    image_3_paths = [os.path.join(data_prefix, right_img_path.replace(cur_id,last_id)), \
                                     os.path.join(data_prefix, right_img_path), \
                                        os.path.join(data_prefix, right_img_path.replace(cur_id,next_id))]
                    dict_i = dict(
                        image_0_paths=[ a.replace("image_02", "image_00") for a in image_2_paths],
                        image_1_paths=[ a.replace("image_02", "image_01") for a in image_2_paths],
                        image_2_paths=image_2_paths,
                        image_3_paths=image_3_paths,
                        gt_poses=gt_poses,
                        K_0 = cam_intrinsics_gray[0],
                        K_1 = cam_intrinsics_gray[1],
                        K_2= cam_intrinsics[0],
                        K_3= cam_intrinsics[1],
                        focal_0=str(focal0),
                        focal_1=str(focal1),
                        focal_2=str(focal2),
                        focal_3=str(focal3),
                        baseline_01=str(baseline01),
                        baseline_23=str(baseline23),
                        seq_len = 2,
                        pseudo_seq = 0,
                        #original_size=(img_height, img_width),
                    )
                    outputs.append(dict_i)
                
    print(f"total {len(outputs)} samples")
    print(f"pre seq {pre_seq} samples")
    print(f"next seq {next_seq} samples")
    print(f"still seq {still_seq} samples")

    return outputs  


def make_test_annotations(image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt',
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_raw_data', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png', selfsup=True ):
    gtpose_ann, raw_imgid_to_odom_imgid = get_pose()
    outputs = []
    pre_seq, next_seq, still_seq = 0,0,0
    with open(image_ann_path, 'r') as f:
        image_anns = f.readlines()
        print("raw data len: ", len(image_anns))
        for img_ann in tqdm(image_anns):
            # e.g. 2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000215.jpg 2011_09_26/2011_09_26_drive_0070_sync/image_03/data/0000000215.jpg
            left_img_path, right_img_path = img_ann.strip().split(' ')
            left_img_path, right_img_path = left_img_path.replace("jpg", "png"), right_img_path.replace("jpg", "png")
            date_id = left_img_path.split('/')[0]
            raw_file_id = left_img_path.split('/')[1]
            img_id = left_img_path.split('/')[-1].split('.')[0]
            # K 
            calib_path = os.path.join(data_prefix, date_id , "calib_cam_to_cam.txt")
            cam_intrinsics = load_kitti_depth_odom_intrinsics(calib_path) # a dict of camera 0, 1, 2, 3, tr 
            cam_intrinsics_gray = load_kitti_depth_odom_intrinsics(calib_path, gray=True)
            focal0, focal1, focal2, focal3, baseline01, baseline23 = get_focal_length_baseline(calib_path, 2)
            
            dict_i = dict()
            gt_poses = []
            if left_img_path in raw_imgid_to_odom_imgid:
                # get the next frame and odometry GT pose
                odom_seqimgid = raw_imgid_to_odom_imgid[left_img_path]
                odom_seqid = int(odom_seqimgid.split('/')[-2])
                odom_imgid = int(odom_seqimgid.split('/')[-1].split('.')[0])
                raw_imgid = int(left_img_path.split('/')[-1].split('.')[0])
                cur_id = '{:0>10}.png'.format(raw_imgid)
                # the last frame
                last_id = '{:0>10}.png'.format(raw_imgid-1)
                if os.path.exists(os.path.join(data_prefix, left_img_path.replace(cur_id, last_id))) and odom_imgid-1 >= 0 and odom_imgid-1 < len(gtpose_ann[odom_seqid]):
                    pre_flag = 1
                else:
                    pre_flag = 0
                # the next frame 
                next_id = '{:0>10}.png'.format(raw_imgid+1)
                if os.path.exists(os.path.join(data_prefix, left_img_path.replace(cur_id, next_id))) and odom_imgid+1 >= 0 and odom_imgid+1 < len(gtpose_ann[odom_seqid]):
                    next_flag = 1
                else:
                    next_flag = 0

                
                if next_flag == 1:
                    next_seq +=1
                    gt_poses = [ gtpose_ann[odom_seqid][odom_imgid],
                                            gtpose_ann[odom_seqid][odom_imgid+1] ]
                    image_2_paths = [os.path.join(data_prefix, left_img_path), os.path.join(data_prefix, left_img_path.replace(cur_id,next_id))]
                    image_3_paths = [os.path.join(data_prefix, right_img_path), os.path.join(data_prefix, right_img_path.replace(cur_id,next_id))]
                    dict_i = dict(
                        image_0_paths=[ a.replace("image_02", "image_00") for a in image_2_paths],
                        image_1_paths=[ a.replace("image_02", "image_01") for a in image_2_paths],
                        image_2_paths=image_2_paths,
                        image_3_paths=image_3_paths,
                        gt_poses=gt_poses,
                        K_0 = cam_intrinsics_gray[0],
                        K_1 = cam_intrinsics_gray[1],
                        K_2= cam_intrinsics[0],
                        K_3= cam_intrinsics[1],
                        focal_0=str(focal0),
                        focal_1=str(focal1),
                        focal_2=str(focal2),
                        focal_3=str(focal3),
                        baseline_01=str(baseline01),
                        baseline_23=str(baseline23),
                        seq_len = 2,
                        pseudo_seq = 0,
                        #original_size=(img_height, img_width),
                    )
                    outputs.append(dict_i)
                else:# there is not corresponding odometry frame
                    still_seq +=1
                    gt_poses = [ np.eye(4), np.eye(4) ]
                    image_2_paths = [os.path.join(data_prefix, left_img_path), os.path.join(data_prefix, left_img_path)]
                    image_3_paths = [os.path.join(data_prefix, right_img_path), os.path.join(data_prefix, right_img_path)]

                    dict_i = dict(
                        image_0_paths=[ a.replace("image_02", "image_00") for a in image_2_paths],
                        image_1_paths=[ a.replace("image_02", "image_01") for a in image_2_paths],
                        image_2_paths=image_2_paths,
                        image_3_paths=image_3_paths,
                        gt_poses=gt_poses,
                        K_0 = cam_intrinsics_gray[0],
                        K_1 = cam_intrinsics_gray[1],
                        K_2= cam_intrinsics[0],
                        K_3= cam_intrinsics[1],
                        focal_0=str(focal0),
                        focal_1=str(focal1),
                        focal_2=str(focal2),
                        focal_3=str(focal3),
                        baseline_01=str(baseline01),
                        baseline_23=str(baseline23),
                        seq_len = 2,
                        pseudo_seq = 1,
                        #original_size=(img_height, img_width),
                    )
                    outputs.append(dict_i)

            else:# there is not corresponding odometry frame
                continue
                still_seq +=1
                gt_poses = [ np.eye(4), np.eye(4) ]
                image_2_paths = [os.path.join(data_prefix, left_img_path), os.path.join(data_prefix, left_img_path)]
                image_3_paths = [os.path.join(data_prefix, right_img_path), os.path.join(data_prefix, right_img_path)]

                dict_i = dict(
                    image_0_paths=[ a.replace("image_02", "image_00") for a in image_2_paths],
                    image_1_paths=[ a.replace("image_02", "image_01") for a in image_2_paths],
                    image_2_paths=image_2_paths,
                    image_3_paths=image_3_paths,
                    gt_poses=gt_poses,
                    K_0 = cam_intrinsics_gray[0],
                    K_1 = cam_intrinsics_gray[1],
                    K_2= cam_intrinsics[0],
                    K_3= cam_intrinsics[1],
                    focal_0=str(focal0),
                    focal_1=str(focal1),
                    focal_2=str(focal2),
                    focal_3=str(focal3),
                    baseline_01=str(baseline01),
                    baseline_23=str(baseline23),
                    seq_len = 2,
                    pseudo_seq = 1,
                    #original_size=(img_height, img_width),
                )
                outputs.append(dict_i)
    print(f"total {len(outputs)} samples")
    print(f"pre seq {pre_seq} samples")
    print(f"next seq {next_seq} samples")
    print(f"still seq {still_seq} samples")
    return outputs  


def make_seqtest_annotations(image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt',
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_raw_data', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png', selfsup=True ):
    gtpose_ann, raw_imgid_to_odom_imgid = get_pose()
    outputs = []
    pre_seq, next_seq, still_seq = 0,0,0
    with open(image_ann_path, 'r') as f:
        image_anns = f.readlines()
        print("raw data len: ", len(image_anns))
        for img_ann in tqdm(image_anns):
            # e.g. 2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000215.jpg 2011_09_26/2011_09_26_drive_0070_sync/image_03/data/0000000215.jpg
            left_img_path, right_img_path = img_ann.strip().split(' ')
            left_img_path, right_img_path = left_img_path.replace("jpg", "png"), right_img_path.replace("jpg", "png")
            date_id = left_img_path.split('/')[0]
            raw_file_id = left_img_path.split('/')[1]
            img_id = left_img_path.split('/')[-1].split('.')[0]
            # K 
            calib_path = os.path.join(data_prefix, date_id , "calib_cam_to_cam.txt")
            cam_intrinsics = load_kitti_depth_odom_intrinsics(calib_path) # a dict of camera 0, 1, 2, 3, tr 
            cam_intrinsics_gray = load_kitti_depth_odom_intrinsics(calib_path, gray=True)
            focal0, focal1, focal2, focal3, baseline01, baseline23 = get_focal_length_baseline(calib_path, 2)
            
            dict_i = dict()
            gt_poses = []
            #if left_img_path in raw_imgid_to_odom_imgid:
                # get the next frame and odometry GT pose
            #    odom_seqimgid = raw_imgid_to_odom_imgid[left_img_path]
               # odom_seqid = int(odom_seqimgid.split('/')[-2])
               #odom_imgid = int(odom_seqimgid.split('/')[-1].split('.')[0])
            raw_imgid = int(left_img_path.split('/')[-1].split('.')[0])
            cur_id = '{:0>10}.png'.format(raw_imgid)
            # the last frame
            last_id = '{:0>10}.png'.format(raw_imgid-1)
            if os.path.exists(os.path.join(data_prefix, left_img_path.replace(cur_id, last_id))):
                pre_flag = 1
            else:
                pre_flag = 0
            # the next frame 
            next_id = '{:0>10}.png'.format(raw_imgid+1)
            if os.path.exists(os.path.join(data_prefix, left_img_path.replace(cur_id, next_id))):
                next_flag = 1
            else:
                next_flag = 0
            if next_flag == 1:
                image_2_paths = [os.path.join(data_prefix, left_img_path), os.path.join(data_prefix, left_img_path.replace(cur_id,next_id))]
                image_3_paths = [os.path.join(data_prefix, right_img_path), os.path.join(data_prefix, right_img_path.replace(cur_id,next_id))]
                dict_i = dict(
                    image_0_paths=[ a.replace("image_02", "image_00") for a in image_2_paths],
                    image_1_paths=[ a.replace("image_02", "image_01") for a in image_2_paths],
                    image_2_paths=image_2_paths,
                    image_3_paths=image_3_paths,
                    #gt_poses=gt_poses,
                    K_0 = cam_intrinsics_gray[0],
                    K_1 = cam_intrinsics_gray[1],
                    K_2= cam_intrinsics[0],
                    K_3= cam_intrinsics[1],
                    focal_0=str(focal0),
                    focal_1=str(focal1),
                    focal_2=str(focal2),
                    focal_3=str(focal3),
                    baseline_01=str(baseline01),
                    baseline_23=str(baseline23),
                    seq_len = 2,
                    pseudo_seq = 0,
                    #original_size=(img_height, img_width),
                )
            else:
                image_2_paths = [os.path.join(data_prefix, left_img_path), os.path.join(data_prefix, left_img_path)]
                image_3_paths = [os.path.join(data_prefix, right_img_path), os.path.join(data_prefix, right_img_path)]

                dict_i = dict(
                    image_0_paths=[ a.replace("image_02", "image_00") for a in image_2_paths],
                    image_1_paths=[ a.replace("image_02", "image_01") for a in image_2_paths],
                    image_2_paths=image_2_paths,
                    image_3_paths=image_3_paths,
                    gt_poses=gt_poses,
                    K_0 = cam_intrinsics_gray[0],
                    K_1 = cam_intrinsics_gray[1],
                    K_2= cam_intrinsics[0],
                    K_3= cam_intrinsics[1],
                    focal_0=str(focal0),
                    focal_1=str(focal1),
                    focal_2=str(focal2),
                    focal_3=str(focal3),
                    baseline_01=str(baseline01),
                    baseline_23=str(baseline23),
                    seq_len = 2,
                    pseudo_seq = 1,
                    #original_size=(img_height, img_width),
                )
            outputs.append(dict_i)

    print(f"total {len(outputs)} samples")
    print(f"pre seq {pre_seq} samples")
    print(f"next seq {next_seq} samples")
    print(f"still seq {still_seq} samples")
    return outputs  

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def make_json(save_annotation_root, name, metas):
    filepath = os.path.join(save_annotation_root, f"{name}" + '.json')
    print('Save to {}'.format(filepath))
    with open(file=filepath, mode='w') as fp:
        json.dump(metas, fp=fp, cls=NumpyEncoder)




def save(save_path, name, image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_train_files.txt', selfsup=True):
    save_annotation_root = os.path.join(save_path,)
    if not os.path.exists(save_annotation_root):
        os.makedirs(save_annotation_root)
    metas = make_annotations(image_ann_path, selfsup=selfsup)

    make_json(save_annotation_root, name, metas)

def save_test(save_path, name, image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt', selfsup=True):
    save_annotation_root = os.path.join(save_path,)
    if not os.path.exists(save_annotation_root):
        os.makedirs(save_annotation_root)
    metas = make_test_annotations(image_ann_path, selfsup=selfsup)

    make_json(save_annotation_root, name, metas)

def save_seqtest(save_path, name, image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt', selfsup=True):
    save_annotation_root = os.path.join(save_path,)
    if not os.path.exists(save_annotation_root):
        os.makedirs(save_annotation_root)
    metas = make_seqtest_annotations(image_ann_path, selfsup=selfsup)

    make_json(save_annotation_root, name, metas)

if __name__ == '__main__':
    #print("cam01 mean std")
    #count_cam01_mean_std()
    #print("cam23 mean std")
    #count_cam23_mean_std()
    save('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_train_len3',\
         image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_train_files.txt')
    #save_seqtest('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_test_seq',\
    #     image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt')
    #save_test('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_test_gtpose',\
    #     image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt')
    
    #save_test('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_test_onlywpose',\
    #     image_ann_path='/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt')
    #save('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_train_clean')
    #save('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_train')
    #save('/home/ziliu/mydata/kitti_depth', 'kitti_eigen_unsup_test', '/home/ziliu/mydata/depth_splits/eigen_monodepth/eigen_test_files.txt')
 
"""
total 35809 samples
pre seq 13209 samples
next seq 13215 samples
still seq 9385 samples
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_train.json
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_train.json
raw data len:  697
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 697/697 [00:00<00:00, 706.04it/s]
total 697 samples
pre seq 0 samples
next seq 99 samples
still seq 598 samples
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_test_gtpose.json
raw data len:  697
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 697/697 [00:01<00:00, 598.92it/s]
total 697 samples
pre seq 0 samples
next seq 0 samples
still seq 0 samples
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_test_seq.json


len ==3 
total 13209 samples
pre seq 0 samples
next seq 0 samples
still seq 0 samples
Save to /home/ziliu/mydata/kitti_depth/kitti_eigen_unsup_train_len3.json

"""
