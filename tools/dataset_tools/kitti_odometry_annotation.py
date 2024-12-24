import numpy as np
import cv2
import json
import os 
from tqdm import tqdm
import glob
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
    idx_odometryID_acc_rawfileID = dict()
    for gtposeidx in range(len(gt_pose_raw_idx)):
        odometry_id, raw_id, start_frameid, end_frameid = gt_pose_raw_idx[gtposeidx].split(' ')
        odometry_id = odometry_id.split(':')[0] # str '00'
        idx_odometryID_acc_rawfileID[raw_id+"_sync"] = odometry_id
    # load all gt abs pose from KITTI odometry GT pose.txt
    gt_abs_poses_acc_eachSEQ = dict() # {odometry_id: [4,4]}
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
            gt_abs_poses_acc_eachSEQ["{:0>2}".format(odometry_idx)] = left_poses_abs
    return gt_abs_poses_acc_eachSEQ, idx_odometryID_acc_rawfileID



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


def make_annotations( test_mode, test_seq=None,
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_odometry', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png' ):
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

    #gt_abs_poses_acc_eachSEQ, idx_odometryID_acc_rawfileID = get_pose()
    if not test_mode:
        seq_ids = ['{:0>2}'.format(i) for i in range(9)]
        #seq_ids.remove('03')
    else:
        seq_ids = ['{:0>2}'.format(test_seq)]

    
    outputs = []
    for seq_id in seq_ids:
        image2_dir = os.path.join(data_prefix, "sequences", seq_id, 'image_2')
        image2s = sorted(glob.glob(os.path.join(image2_dir, '*.png')))
        with open(os.path.join(data_prefix, "pose_GT", f'{seq_id}.txt'), 'r') as f:
            gt_poses = f.readlines()
        calib_path = os.path.join(data_prefix, "sequences", seq_id , "calib.txt")
        with open(calib_path, 'r') as f:
            calibs = f.readlines()
            K_0 = calibs[0].strip().split(' ')[1:]
            K_0 = ' '.join(K_0)
            K_1 = calibs[1].strip().split(' ')[1:]
            K_1 = ' '.join(K_1)
            focal_0 = float(K_0.split(' ')[0])
            focal_1 = float(K_1.split(' ')[0])
            # cam 2 is left of camera 0  -6cm
            # cam 3 is to the right  +54cm
            b0 = float(K_0.split(' ')[3]) / -float(K_0.split(' ')[0])
            b1 = float(K_1.split(' ')[3]) / -float(K_1.split(' ')[0])
            baseline_01 = b1-b0
            K_2 = calibs[2].strip().split(' ')[1:]
            K_2 = ' '.join(K_2)
            K_3 = calibs[3].strip().split(' ')[1:]
            K_3 = ' '.join(K_3)
            focal_2 = float(K_2.split(' ')[0])
            focal_3 = float(K_3.split(' ')[0])
            # cam 2 is left of camera 0  -6cm
            # cam 3 is to the right  +54cm
            b2 = float(K_2.split(' ')[3]) / -float(K_2.split(' ')[0])
            b3 = float(K_3.split(' ')[3]) / -float(K_3.split(' ')[0])
            baseline_23 = b3-b2


        assert len(image2s) == len(gt_poses), f"image0 {len(image2s)}, gt poses {len(gt_poses)}"
        num = len(image2s)
        
        for refid, curid in tqdm(zip(range(0,num-1), range(1,num))):
            left_image_paths = [image2s[refid], image2s[curid]]
            right_image_paths = [image2s[refid].replace("image_2", "image_3"), image2s[curid].replace("image_2", "image_3")]
            dict_i = dict(
                #image_0_paths=[image2s[refid], image2s[curid]],
                #image_1_paths=[image2s[refid].replace("image_0", "image_1"), image2s[curid].replace("image_0", "image_1")],
                image_2_paths=[image2s[refid], image2s[curid]],
                image_3_paths=[image2s[refid].replace("image_2", "image_3"), image2s[curid].replace("image_2", "image_3")],
                #depth_0_paths= None,
                #depth_1_paths= None,
                #depth_2_paths=left_depth_paths if not selfsup else None,
                #depth_3_paths=right_depth_paths if not selfsup else None,
                gt_poses=[gt_poses[refid].strip(), gt_poses[curid].strip()],
                #K_0 = K_0,
                #K_1 = K_1,
                K_2 = K_2,
                K_3 = K_3,
                #K_2= " ".join([str(a)  for a in cam_intrinsics[0].reshape(-1)]),
                #K_3= " ".join([str(a)  for a in cam_intrinsics[1].reshape(-1)]),
                #focal_0=str(focal_0),
                #focal_1=str(focal_1),
                focal_2=str(focal_2),
                focal_3=str(focal_3),
                #focal_2=str(focal2),
                #focal_3=str(focal3),
                #baseline_01=str(baseline_01),
                baseline_23=str(baseline_23),
                seq_len = 2,
                #pseudo_seq = pseudo_seq,
                #original_size=(img_height, img_width),
            )
            outputs.append(dict_i)
    
    return outputs  

def make_annotations_3frames( test_mode, test_seq=None,
               data_prefix= '/data/acentauri/user/ziliu/data/kitti_odometry', filename_tmpl='{:0>10}.png', 
                 depth_filename_tmpl='{:0>10}.png' ):
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

    #gt_abs_poses_acc_eachSEQ, idx_odometryID_acc_rawfileID = get_pose()
    if not test_mode:
        seq_ids = ['{:0>2}'.format(i) for i in range(9)]
        #seq_ids.remove('03')
    else:
        seq_ids = ['{:0>2}'.format(test_seq)]

    
    outputs = []
    for seq_id in seq_ids:
        image2_dir = os.path.join(data_prefix, "sequences", seq_id, 'image_2')
        image2s = sorted(glob.glob(os.path.join(image2_dir, '*.png')))
        with open(os.path.join(data_prefix, "pose_GT", f'{seq_id}.txt'), 'r') as f:
            gt_poses = f.readlines()
        calib_path = os.path.join(data_prefix, "sequences", seq_id , "calib.txt")
        with open(calib_path, 'r') as f:
            calibs = f.readlines()
            K_0 = calibs[0].strip().split(' ')[1:]
            K_0 = ' '.join(K_0)
            K_1 = calibs[1].strip().split(' ')[1:]
            K_1 = ' '.join(K_1)
            focal_0 = float(K_0.split(' ')[0])
            focal_1 = float(K_1.split(' ')[0])
            # cam 2 is left of camera 0  -6cm
            # cam 3 is to the right  +54cm
            b0 = float(K_0.split(' ')[3]) / -float(K_0.split(' ')[0])
            b1 = float(K_1.split(' ')[3]) / -float(K_1.split(' ')[0])
            baseline_01 = b1-b0
            K_2 = calibs[2].strip().split(' ')[1:]
            K_2 = ' '.join(K_2)
            K_3 = calibs[3].strip().split(' ')[1:]
            K_3 = ' '.join(K_3)
            focal_2 = float(K_2.split(' ')[0])
            focal_3 = float(K_3.split(' ')[0])
            # cam 2 is left of camera 0  -6cm
            # cam 3 is to the right  +54cm
            b2 = float(K_2.split(' ')[3]) / -float(K_2.split(' ')[0])
            b3 = float(K_3.split(' ')[3]) / -float(K_3.split(' ')[0])
            baseline_23 = b3-b2


        assert len(image2s) == len(gt_poses), f"image0 {len(image2s)}, gt poses {len(gt_poses)}"
        num = len(image2s)
        
        for preid, refid, curid in tqdm(zip(range(0,num-2), range(1,num-1), range(2,num))):
            left_image_paths = [image2s[preid], image2s[refid], image2s[curid]]
            right_image_paths = [image2s[preid].replace("image_2", "image_3"), image2s[refid].replace("image_2", "image_3"), image2s[curid].replace("image_2", "image_3")]
            dict_i = dict(
                #image_0_paths=[image2s[refid], image2s[curid]],
                #image_1_paths=[image2s[refid].replace("image_0", "image_1"), image2s[curid].replace("image_0", "image_1")],
                image_2_paths=[image2s[preid], image2s[refid], image2s[curid]],
                image_3_paths=[image2s[preid].replace("image_2", "image_3"), image2s[refid].replace("image_2", "image_3"), image2s[curid].replace("image_2", "image_3")],
                #depth_0_paths= None,
                #depth_1_paths= None,
                #depth_2_paths=left_depth_paths if not selfsup else None,
                #depth_3_paths=right_depth_paths if not selfsup else None,
                gt_poses=[gt_poses[preid].strip(), gt_poses[refid].strip(), gt_poses[curid].strip()],
                #K_0 = K_0,
                #K_1 = K_1,
                K_2 = K_2,
                K_3 = K_3,
                #K_2= " ".join([str(a)  for a in cam_intrinsics[0].reshape(-1)]),
                #K_3= " ".join([str(a)  for a in cam_intrinsics[1].reshape(-1)]),
                #focal_0=str(focal_0),
                #focal_1=str(focal_1),
                focal_2=str(focal_2),
                focal_3=str(focal_3),
                #focal_2=str(focal2),
                #focal_3=str(focal3),
                #baseline_01=str(baseline_01),
                baseline_23=str(baseline_23),
                seq_len = 3,
                #pseudo_seq = pseudo_seq,
                #original_size=(img_height, img_width),
            )
            outputs.append(dict_i)
    
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




def save(save_path, name,  test_mode, test_seq=None, frame3=False):
    save_annotation_root = os.path.join(save_path,)
    if not os.path.exists(save_annotation_root):
        os.makedirs(save_annotation_root)
    if frame3:
        metas = make_annotations_3frames(test_mode, test_seq)
    else:
        metas = make_annotations(test_mode, test_seq)

    make_json(save_annotation_root, name, metas)


if __name__ == '__main__':
    #save('/home/ziliu/mydata/kitti_odometry', 'kitti_odometry_train_3frame', test_mode=False, frame3=True)
    #save('/home/ziliu/mydata/kitti_odometry', 'kitti_odometry_train', test_mode=False)
    #save('/home/ziliu/mydata/kitti_odometry', 'kitti_odometry_test_09', test_mode=True, test_seq=9)
    save('/home/ziliu/mydata/kitti_odometry', 'kitti_odometry_test_10', test_mode=True, test_seq=10)
    save('/home/ziliu/mydata/kitti_odometry', 'kitti_odometry_test_07', test_mode=True, test_seq=7)
    save('/home/ziliu/mydata/kitti_odometry', 'kitti_odometry_test_08', test_mode=True, test_seq=8)

"""
Save to /home/ziliu/mydata/kitti_odometry/kitti_odometry_train_3frame.json

Save to /home/ziliu/mydata/kitti_odometry/kitti_odometry_train.json
1590it [00:00, 240634.46it/s]
Save to /home/ziliu/mydata/kitti_odometry/kitti_odometry_test_09.json
1200it [00:00, 239378.14it/s]
Save to /home/ziliu/mydata/kitti_odometry/kitti_odometry_test_10.json

"""
