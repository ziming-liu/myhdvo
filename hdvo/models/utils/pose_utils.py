'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-07-07 15:42:22
LastEditors: Ziming Liu
LastEditTime: 2023-07-07 15:57:40
'''
import numpy as np

def pose_relative2absolute(relative_poses, first_abs_pose=np.eye(4)):
    '''
    description: transfrom relative camera pose into absolute camera pose
    parameter: {reltive_poses: [list], camera poses from the reference to the current, i.e. rTc}
    return: {absolute_poses: [list]}
    '''            
    abs_poses = [first_abs_pose, ]
    num = len(relative_poses)
    #print(f"FUNC pose_relative2absolute: \n given {num} relative poses, init first camera pose with \n____\n {first_abs_pose}. \n ___")
    ref_abs_pose = first_abs_pose
    for p_idx in range(num):
        cTr = relative_poses[p_idx]
        cur_abs_pose = np.matmul(ref_abs_pose, np.linalg.inv(cTr))
        theta = -0.5*np.pi
        y_trans_ = np.reshape(np.array([np.cos(theta), 0, -np.sin(theta), 0,1,0, np.sin(theta), 0, np.cos(theta)]), (3,3))
        y_trans = np.eye(4)
        y_trans[:3,:3] = y_trans_
        theta = 0.5*np.pi
        z_trans_ =np.reshape(np.array([np.cos(theta), np.sin(theta), 0, -np.sin(theta), np.cos(theta), 0, 0, 0, 1]), (3,3))
        z_trans = np.eye(4)
        z_trans[:3,:3] = z_trans_
        #cur_abs_pose = np.matmul(cur_abs_pose, np.linalg.inv(y_trans))
        #cur_abs_pose = np.matmul(cur_abs_pose, np.linalg.inv(z_trans))
        
        abs_poses.append(cur_abs_pose)
        ref_abs_pose = cur_abs_pose
    #print(f"FUNC pose_relative2absolute: output {len(abs_poses)} absolute poses.")
    return abs_poses