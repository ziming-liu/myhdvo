# Copyright (C) Huangying Zhan 2019. All rights reserved.

import copy
from inspect import trace
from unittest import result
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import os
from glob import glob


def align_trajectory_sim3(pred_xyz, gt_xyz):
    """
    Align predicted trajectory to ground truth using Sim3 alignment (scale + rotation + translation).
    This is similar to Umeyama alignment.
    
    Args:
        pred_xyz: Predicted trajectory (N, 3)
        gt_xyz: Ground truth trajectory (N, 3)
        
    Returns:
        aligned_pred_xyz: Aligned predicted trajectory
        scale: Scale factor
        R: Rotation matrix
        t: Translation vector
    """
    # Ensure same length
    min_len = min(len(pred_xyz), len(gt_xyz))
    pred_xyz = pred_xyz[:min_len]
    gt_xyz = gt_xyz[:min_len]
    
    # Compute centroids
    pred_centroid = np.mean(pred_xyz, axis=0)
    gt_centroid = np.mean(gt_xyz, axis=0)
    
    # Center the trajectories
    pred_centered = pred_xyz - pred_centroid
    gt_centered = gt_xyz - gt_centroid
    
    # Compute scale
    pred_scale = np.sqrt(np.mean(np.sum(pred_centered**2, axis=1)))
    gt_scale = np.sqrt(np.mean(np.sum(gt_centered**2, axis=1)))
    scale = gt_scale / pred_scale if pred_scale > 0 else 1.0
    
    # Scale the prediction
    pred_scaled = pred_centered * scale
    
    # Compute rotation using SVD
    H = pred_scaled.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    pred_rotated = (R @ pred_scaled.T).T
    
    # Compute translation
    t = gt_centroid - pred_centroid * scale
    
    # Final aligned trajectory
    aligned_pred_xyz = pred_rotated + gt_centroid
    
    return aligned_pred_xyz, scale, R, t

def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y)/np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


class KittiEvalOdom():
    """Evaluate odometry result
    Usage example:
        vo_eval = KittiEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """
    def __init__(self, dataset_type='kitti'):
        """
        Args:
            dataset_type (str): 'kitti' or 'vkitti2'
        """
        # if dataset_type.lower() in ['vkitti2', 'vkitti']:
        #     # VKitti2 trajectory lengths: 711m, 254m, 51.9m, 113m, 332m
        #     # Using shorter intervals to evaluate shorter sequences effectively
        #     self.lengths = [5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700]
        # else:
        #     # Original KITTI odometry evaluation lengths
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        
        self.num_lengths = len(self.lengths)
        self.dataset_type = dataset_type

    def load_poses_from_txt(self, file_name):
        """Load poses from txt (KITTI format)
        Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)

        Args:
            file_name (str): txt file path
        Returns:
            poses (dict): {idx: 4x4 array}
        """
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ") if i!=""]
            withIdx = len(line_split) == 13
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectory_distances(self, poses):
        """Compute distance for each pose w.r.t frame-0
        Args:
            poses (dict): {idx: 4x4 array}
        Returns:
            dist (float list): distance of each pose w.r.t frame-0
        """
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))
        return dist

    def rotation_error(self, pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            rot_error (float): rotation error
        """
        
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        #print("trace R", a+b+c)
        d = 0.5*(a+b+c-1.0)
        if np.min(d)<-1.0 or np.max(d)>1.0:
            # if trace(R) > 3, normlized R, compute again
            u,s,vt = np.linalg.svd(pose_error[:3,:3])
            #print("R >> ", pose_error[:3,:3])
            norm_r = np.dot(u, vt)
            #print("norm_r >> ", norm_r)
            pose_error[:3,:3] = norm_r
            a = pose_error[0, 0]
            b = pose_error[1, 1]
            c = pose_error[2, 2]
            #print("trace R", a+b+c)
            d = 0.5*(a+b+c-1.0)
            #rot_error = np.arccos(d)

        rot_error = np.arccos(d)
        #print(rot_error)
        #print("d>> ",d)
        ##print("arcos d >> ",np.arccos(d))
        #input("Press Enter to continue...")
        return rot_error

    def translation_error(self, pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            trans_error (float): translation error
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx**2+dy**2+dz**2)
        return trans_error

    def last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame (index) that away from the first_frame with
        the required distance
        Args:
            dist (float list): distance of each pose w.r.t frame-0
            first_frame (int): start-frame index
            length (float): required distance
        Returns:
            i (int) / -1: end-frame index. if not found return -1
        """
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """calculate sequence error
        Args:
            poses_gt (dict): {idx: 4x4 array}, ground truth poses
            poses_result (dict): {idx: 4x4 array}, predicted poses
        Returns:
            err (list list): [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        """
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10

        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(
                                        dist, first_frame, len_
                                        )

                # Continue if sequence not long enough
                if last_frame == -1 or \
                        not(last_frame in poses_result.keys()) or \
                        not(first_frame in poses_result.keys()):
                    continue

                # compute rotational and translational errors
                pose_delta_gt = np.dot(
                                    np.linalg.inv(poses_gt[first_frame]),
                                    poses_gt[last_frame]
                                    )
                pose_delta_result = np.dot(
                                        np.linalg.inv(poses_result[first_frame]),
                                        poses_result[last_frame]
                                        )
                pose_error = np.dot(
                                np.linalg.inv(pose_delta_result),
                                pose_delta_gt
                                )
                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # compute speed
                num_frames = last_frame - first_frame + 1.0
                speed = len_/(0.1*num_frames)

                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def save_sequence_errors(self, err, file_name):
        """Save sequence error
        Args:
            err (list list): error information
            file_name (str): txt file for writing errors
        """
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def compute_overall_err(self, seq_err):
        """Compute average translation & rotation errors
        Args:
            seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
                - r_err (float): rotation error
                - t_err (float): translation error
        Returns:
            ave_t_err (float): average translation error
            ave_r_err (float): average rotation error
        """
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)
        #print("seqlen >> ", seq_len)
        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        else:
            return 0, 0

    def plot_trajectory(self, poses_gt, poses_results, seq, given_plot_keys=None):
        """Plot trajectory for both GT and prediction
        Args:
            poses_gt (dict): {idx: 4x4 array}; ground truth poses
            poses_result (dict): {idx: 4x4 array}; predicted poses
            seq (int): sequence index.
        """
        plot_keys = []
        if given_plot_keys is None:
            #one pose result
            if len(poses_results)==1:
                plot_keys = ["Ours"]
            else:
                plot_keys = []
                for m in range(len(poses_results)):
                    plot_keys.append("Ours{}".format(m+1))
        else:
            plot_keys = given_plot_keys.copy()
        if len(plot_keys) < len(poses_results):
            print("there is no enough keys(model names) for the pose resutls")
            raise ValueError
        if len(plot_keys) > len(poses_results):
            print("there is too many keys(model names) for the pose resutls")
            raise ValueError
        assert len(plot_keys) == len(poses_results), "same len"
        fontsize_ = 33

        poses_dict = {}
        for idx in range(len(plot_keys)):
            poses_dict[plot_keys[idx]] = poses_results[idx]
        plot_keys.append("Ground Truth")
        poses_dict["Ground Truth"] = poses_gt
        

        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_aspect('equal')
        
        for key in plot_keys:
            pos_xz = []
            frame_idx_list = sorted(poses_dict[key].keys())
            for frame_idx in frame_idx_list:
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3],  pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0],  pos_xz[:, 1], label=key)
        x_min = min(pos_xz[:, 0])
        x_max = max(pos_xz[:, 0])
        x_mid = int((x_max+x_min)/2)
        z_min = min(pos_xz[:, 1])
        z_max = max(pos_xz[:, 1])
        z_mid = int((z_max+z_min)/2)
        x_len = x_max - x_min
        z_len = z_max - z_min
        half_max_len = int(max(x_len, z_len)/2)

        plt.legend(loc="upper left", prop={'size': fontsize_})
        plt.xticks(np.arange(int(x_mid-half_max_len),int(x_mid+half_max_len),100), fontsize=fontsize_)
        plt.yticks(np.arange(int(z_mid-half_max_len), int(z_mid+half_max_len),100), fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        if isinstance(seq, int):
            png_title = "sequence_{:02}".format(seq)
        else:
            png_title = "sequence_{}".format(seq)
        fig_pdf = self.plot_path_dir + "/" + png_title + ".pdf"
        plt.savefig(fig_pdf, bbox_inches=None, pad_inches=0,dpi=200)
        plt.close(fig)

    def plot_error(self, avg_segment_errs, seq):
        """Plot per-length error
        Args:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
            seq (int): sequence index.
        """
        # Translation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][0] * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Translation Error")
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        seq_str = "{:02}".format(seq) if isinstance(seq, int) else str(seq)
        fig_pdf = self.plot_error_dir + "/trans_err_{}.pdf".format(seq_str)
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Rotation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][1] / np.pi * 180 * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Rotation Error")
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = self.plot_error_dir + "/rot_err_{}.pdf".format(seq_str)
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_segment_error(self, seq_errs):
        """This function calculates average errors for different segment.
        Args:
            seq_errs (list list): list of errs; [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        Returns:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}    
        """

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def compute_ATE(self, gt, pred):
        """Compute RMSE of ATE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        errors = []
        idx_0 = list(pred.keys())[0]
        gt_0 = gt[idx_0]
        pred_0 = pred[idx_0]

        for i in pred:
            # cur_gt = np.linalg.inv(gt_0) @ gt[i]
            cur_gt = gt[i]
            gt_xyz = cur_gt[:3, 3] 

            # cur_pred = np.linalg.inv(pred_0) @ pred[i]
            cur_pred = pred[i]
            pred_xyz = cur_pred[:3, 3]

            align_err = gt_xyz - pred_xyz

            # print('i: ', i)
            # print("gt: ", gt_xyz)
            # print("pred: ", pred_xyz)
            # input("debug")
            errors.append(np.sqrt(np.sum(align_err ** 2)))
        ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
        return ate
    
    def compute_RPE(self, gt, pred):
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i+1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i+1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel
            
            trans_errors.append(self.translation_error(rel_err))
            rot_errors.append(self.rotation_error(rel_err))
        # rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
        # rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        
        return rpe_trans, rpe_rot

    def scale_optimization(self, gt, pred):
        """ Optimize scaling factor
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            new_pred (4x4 array dict): predicted poses after optimization
        """
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in pred:
            pose_pred = pred[i]
            pose_ref = gt[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated
    
    def write_result(self, f, seq, errs):
        """Write result into a txt file
        Args:
            f (IOWrapper)
            seq (int): sequence number
            errs (list): [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot]
        """
        ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot = errs
        lines = []
        lines.append("Sequence: \t {} \n".format(seq) )
        lines.append("Trans. err. (%): \t {:.3f} \n".format(ave_t_err*100))
        lines.append("Rot. err. (deg/100m): \t {:.3f} \n".format(ave_r_err/np.pi*180*100))
        lines.append("ATE (m): \t {:.3f} \n".format(ate))
        lines.append("RPE (m): \t {:.3f} \n".format(rpe_trans))
        lines.append("RPE (deg): \t {:.3f} \n\n".format(rpe_rot * 180 /np.pi))
        for line in lines:
            f.writelines(line)


    def eval(self, gt_dir, result_dir, 
                alignment=None,
                seqs=None,
                plot_keys=["Ours"]):
        """Evaulate required/available sequences
        Args:
            gt_dir (str): ground truth poses txt files directory
            result_dir (str): pose predictions txt files directory
            alignment (str): if not None, optimize poses by
                - scale: optimize scale factor for trajectory alignment and evaluation
                - scale_7dof: optimize 7dof for alignment and use scale for trajectory evaluation
                - 7dof: optimize 7dof for alignment and evaluation
                - 6dof: optimize 6dof for alignment and evaluation
            seqs (list/None):
                - None: Evalute all available seqs in result_dir
                - list: list of sequence indexs to be evaluated
        """
        seq_list = ["{:02}".format(i) for i in range(0, 11)]

        # Initialization
        self.gt_dir = gt_dir
        ave_t_errs = []
        ave_r_errs = []
        seq_ate = []
        seq_rpe_trans = []
        seq_rpe_rot = []

        # Create result directory
        error_dir = result_dir + "/errors"
        self.plot_path_dir = result_dir + "/plot_path"
        self.plot_error_dir = result_dir + "/plot_error"
        result_txt = os.path.join(result_dir, "result.txt")
        f = open(result_txt, 'w')

        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
        if not os.path.exists(self.plot_path_dir):
            os.makedirs(self.plot_path_dir)
        if not os.path.exists(self.plot_error_dir):
            os.makedirs(self.plot_error_dir)

        # Create evaluation list
        if seqs is None:
            available_seqs = sorted(glob(os.path.join(result_dir, "*.txt")))
            self.eval_seqs = [int(i[-6:-4]) for i in available_seqs if i[-6:-4] in seq_list]
        else:
            self.eval_seqs = seqs

        # evaluation
        for seqi in self.eval_seqs:
            self.all_file_names = seqi.split(',')
            self.all_file_names = [ni for ni in self.all_file_names]
            
            # Support both numeric and string sequence IDs
            first_seq_name = str(self.all_file_names[0])
            try:
                i = int(first_seq_name)
                self.cur_seq = '{:02}'.format(i)
                file_name = '{:02}.txt'.format(i)
                seq_id = i  # For plotting
            except ValueError:
                # String sequence ID (e.g., "Scene99")
                self.cur_seq = first_seq_name
                file_name = '{}.txt'.format(first_seq_name)
                seq_id = first_seq_name  # For plotting

            all_pose_results = []
            for k in range(len(self.all_file_names)):
                k_file_name = '{}.txt'.format(self.all_file_names[k])
                
                poses_result = self.load_poses_from_txt(result_dir+"/"+k_file_name)
                poses_gt = self.load_poses_from_txt(self.gt_dir + "/" + file_name)
                self.result_file_name = result_dir+"/"+k_file_name
                print("result file name ", self.result_file_name)
                print("gt file name ", self.gt_dir + "/" + file_name)

                # Pose alignment to first frame
                idx_0 = sorted(list(poses_result.keys()))[0]
                pred_0 = poses_result[idx_0]
                gt_0 = poses_gt[idx_0]
                for cnt in poses_result:
                    poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
                    poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

                if alignment == "scale":
                    poses_result = self.scale_optimization(poses_gt, poses_result)
                elif alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
                    # get XYZ
                    xyz_gt = []
                    xyz_result = []
                    for cnt in poses_result:
                        xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
                        xyz_result.append([poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
                    xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                    xyz_result = np.asarray(xyz_result).transpose(1, 0)

                    r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment!="6dof")

                    align_transformation = np.eye(4)
                    align_transformation[:3:, :3] = r
                    align_transformation[:3, 3] = t
                    
                    for cnt in poses_result:
                        poses_result[cnt][:3, 3] *= scale
                        if alignment=="7dof" or alignment=="6dof":
                            poses_result[cnt] = align_transformation @ poses_result[cnt]

                # add all pose results
                
                all_pose_results.append(poses_result)

                # compute sequence errors
                seq_err = self.calc_sequence_errors(poses_gt, poses_result)
                self.save_sequence_errors(seq_err, error_dir + "/" + file_name)

                # Compute segment errors
                avg_segment_errs = self.compute_segment_error(seq_err)

                # compute overall error
                ave_t_err, ave_r_err = self.compute_overall_err(seq_err)
                print("Sequence: ", k_file_name)
                print("Translational error (%): ", ave_t_err*100)
                print("Rotational error (deg/100m): ", ave_r_err/np.pi*180*100)
                ave_t_errs.append(ave_t_err)
                ave_r_errs.append(ave_r_err)

                # Compute ATE
                ate = self.compute_ATE(poses_gt, poses_result)
                seq_ate.append(ate)
                print("ATE (m): ", ate)

                # Compute RPE
                rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
                seq_rpe_trans.append(rpe_trans)
                seq_rpe_rot.append(rpe_rot)
                print("RPE (m): ", rpe_trans)
                print("RPE (deg): ", rpe_rot * 180 /np.pi, " rad >> ", rpe_rot)

                
                self.plot_error(avg_segment_errs, self.all_file_names[k])

                # Save result summary
                self.write_result(f, self.all_file_names[k], [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot])

            # Plotting all trajectories into one fig. onece.
            self.plot_trajectory(poses_gt, all_pose_results, seq_id, plot_keys)
            
            
        f.close()    

        print("-------------------- For Copying ------------------------------")
        for i in range(len(ave_t_errs)):
            print("{0:.2f}".format(ave_t_errs[i]*100))
            print("{0:.2f}".format(ave_r_errs[i]/np.pi*180*100))
            print("{0:.2f}".format(seq_ate[i]))
            print("{0:.3f}".format(seq_rpe_trans[i]))
            print("{0:.3f}".format(seq_rpe_rot[i] * 180 / np.pi))
