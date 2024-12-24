'''
Author: Ziming Liu
Date: 2022-06-16 11:24:58
LastEditors: Ziming Liu
LastEditTime: 2023-07-15 14:26:33
Description: operation of saveing and loading depth maps of a sequence, for testing results.
Dependent packages: cv2, numpy, 
'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def colored_depthmap(depth, d_min=1, d_max=1000):
    gt_height, gt_width = depth.shape
    #crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
    #                                0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
    #depth = depth[int(crop[0]):int(crop[1]), int(crop[2]):int(crop[3])]
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    colored_depth = (depth_relative * 255)#.astype('uint8')
    
    #depth = (depth_relative*255).astype(np.uint8)
    #colored_depth = cv2.applyColorMap(depth,cv2.COLORMAP_SUMMER   )
    #return 255 * cmap(depth_relative)[:,:,:3] # HWC
    return colored_depth

def save_depth_maps(dataset_type, test_seq_id, pred_depth, work_dir, epoch, stereo_view="left", ifgtdepth=False, min_depth =  1e-3, max_depth=2000,first_frame_id=0 ):
    '''
    description: 
    parameters: dataset_type: "KittiDepthStereoDataset""KittiStereoMatchingDataset" are static images dataset. we give a pseudo seqID 99.
                pred_dpeth: prediction depth maps

    return: {*}
    '''    
    num_samples = len(pred_depth)
    if dataset_type == "KittiDepthStereoDataset" or dataset_type == "KittiStereoMatchingDataset":
        test_seq_id = "99"
    else:
        test_seq_id = test_seq_id

    depth_dir = os.path.join(work_dir,f"{epoch}_pred_depths_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
    vis_depth_dir = os.path.join(work_dir,f"{epoch}_pred_depths_color_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
    txt_depth_dir =  os.path.join(work_dir,f"{epoch}_pred_depths_txt_{stereo_view}_{dataset_type}_seq"+ test_seq_id)    
    if ifgtdepth:
        depth_dir = os.path.join(work_dir,f"{epoch}_gt_depths_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
        vis_depth_dir = os.path.join(work_dir,f"{epoch}_gt_depths_color_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
        txt_depth_dir =  os.path.join(work_dir,f"{epoch}_gt_depths_txt_{stereo_view}_{dataset_type}_seq"+ test_seq_id)    
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    if not os.path.exists(vis_depth_dir):
        os.makedirs(vis_depth_dir)
    if not os.path.exists(txt_depth_dir):
        os.makedirs(txt_depth_dir)
   
    print("saving {} pred depths into {}".format(num_samples, depth_dir))
    #print("saving {} pred depths into txt {}".format(num_samples, txt_depth_dir))

    for i in range(len(pred_depth)):
        # save depth into txt with meters 
        #save_txt_path = os.path.join(txt_depth_dir, '{:0>6}.txt'.format(+i))
        #np.set_printoptions(suppress=True)
        #np.set_printoptions(precision=6)
        #np.savetxt(save_txt_path, pred_depth[i], fmt='%.06f')
        # follow the operations in https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/utils/evaluate_kitti.py
        
        # save original pred depth map
        #pred_depth[i] = np.clip(pred_depth[i], min_depth, max_depth)*100
        pred_depth[i] = pred_depth[i]*100
        pred_depth[i] = pred_depth[i].astype(np.uint16) # meter to centen meter
        save_img_path = os.path.join(depth_dir, '{:0>6}.png'.format(first_frame_id+i))
        if len(pred_depth[i].shape)==3:
            pred_depth[i] = np.squeeze(pred_depth[i], 0) 
        elif len(pred_depth[i].shape)==2:
            pass
        else:
            print("shape: ", pred_depth[i].shape)
            raise ValueError
        cv2.imwrite(save_img_path, pred_depth[i].squeeze() ) # save depth into png 

        # save visualization depths 
        save_vis_path = os.path.join(vis_depth_dir, '{:0>6}.png'.format(first_frame_id+i))
        di = 300 / (pred_depth[i]+1e-5) # to disparity map, this sacle is not true
        depth_np_color = colored_depthmap(di)
        vmax = np.percentile(depth_np_color, 95)
        plt.imsave(save_vis_path, depth_np_color, format='png', cmap='magma', vmax=vmax)


def save_kittistereo_disp_maps(dataset_type, test_seq_id, pred_depth, imgs_ids, work_dir, epoch, stereo_view="left", ifgtdepth=False, min_depth =  1e-3, max_depth=2000, ):
    '''
    description: 
    parameters: dataset_type: "KittiDepthStereoDataset""KittiStereoMatchingDataset" are static images dataset. we give a pseudo seqID 99.
                pred_dpeth: prediction depth maps

    return: {*}
    '''    
    num_samples = len(pred_depth)
    if dataset_type == "KittiDepthStereoDataset" or dataset_type == "KittiStereoMatchingDataset":
        test_seq_id = "99"
    else:
        test_seq_id = test_seq_id

    depth_dir = os.path.join(work_dir,f"{epoch}_pred_depths_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
    vis_depth_dir = os.path.join(work_dir,f"{epoch}_pred_depths_color_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
    txt_depth_dir =  os.path.join(work_dir,f"{epoch}_pred_depths_txt_{stereo_view}_{dataset_type}_seq"+ test_seq_id)    
    if ifgtdepth:
        depth_dir = os.path.join(work_dir,f"{epoch}_gt_depths_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
        vis_depth_dir = os.path.join(work_dir,f"{epoch}_gt_depths_color_{stereo_view}_{dataset_type}_seq"+ test_seq_id)
        txt_depth_dir =  os.path.join(work_dir,f"{epoch}_gt_depths_txt_{stereo_view}_{dataset_type}_seq"+ test_seq_id)    
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    if not os.path.exists(vis_depth_dir):
        os.makedirs(vis_depth_dir)
    if not os.path.exists(txt_depth_dir):
        os.makedirs(txt_depth_dir)
   
    print("saving {} pred depths into {}".format(num_samples, depth_dir))
    #print("saving {} pred depths into txt {}".format(num_samples, txt_depth_dir))

    for i in range(len(pred_depth)):
        # save depth into txt with meters 
        #save_txt_path = os.path.join(txt_depth_dir, '{:0>6}.txt'.format(+i))
        #np.set_printoptions(suppress=True)
        #np.set_printoptions(precision=6)
        #np.savetxt(save_txt_path, pred_depth[i], fmt='%.06f')
        # follow the operations in https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/utils/evaluate_kitti.py
        
        # save original pred depth map
        #pred_depth[i] = np.clip(pred_depth[i], min_depth, max_depth)*100
        pred_depth[i] = pred_depth[i]*256
        pred_depth[i] = pred_depth[i].astype(np.uint16) # meter to centen meter
        save_img_path = os.path.join(depth_dir, '{}.png'.format(imgs_ids[i].split("/")[-1].split(".")[0]))
        if len(pred_depth[i].shape)==3:
            pred_depth[i] = np.squeeze(pred_depth[i], 0) 
        elif len(pred_depth[i].shape)==2:
            pass
        else:
            print("shape: ", pred_depth[i].shape)
            raise ValueError
        rgb = cv2.imread(imgs_ids[i])
        h,w,c = rgb.shape
        padh, padw = pred_depth[i].shape
        cv2.imwrite(save_img_path, pred_depth[i].squeeze()[(padh-h)//2:(padh-h)//2+h,(padw-w)//2:(padw-w)//2+w] ) # save depth into png 

        # save visualization depths 
        save_vis_path = os.path.join(vis_depth_dir, '{}.png'.format(imgs_ids[i].split("/")[-1].split(".")[0]))
        di = 300 / (pred_depth[i]+1e-5) # to disparity map, this sacle is not true
        depth_np_color = colored_depthmap(di)
        depth_np_color = depth_np_color[(padh-h)//2:(padh-h)//2+h,(padw-w)//2:(padw-w)//2+w]
        vmax = np.percentile(depth_np_color, 95)
        plt.imsave(save_vis_path, depth_np_color, format='png', cmap='magma', vmax=vmax)

