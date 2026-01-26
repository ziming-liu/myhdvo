'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-04-27 16:49:43
LastEditors: Ziming Liu
LastEditTime: 2023-06-15 17:05:32
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from hdvo.core.visulization import vis_depth_tensor,vis_img_tensor
import numpy as np
from .grid_sample_hessian import grid_sample_hessian
#try:
# from hdvo.models.utils.cuda_gridsample_grad2 import grid_sample_2d, grid_sample_2d_esm, grid_sample_2d_fc, grid_sample_2d_ic
#except:
#    print("Warning: cuda_gridsample_grad2 is not compiled, please compile it first. load gcc, cmake")


def stereo_warp(img, disp, direction, mode='bilinear',padding_mode="zeros",align_corners=False):
    '''
    Description: 
    Args:
        direction: 'l2r' warp the left to the right or 'r2l' warp the right to the left
    Returns:: warped iamge 
    '''
    if direction=='r2l':
        return stereo_warp_r2l(img, disp, mode,padding_mode,align_corners)
    elif direction=='l2r':
        return stereo_warp_l2r(img, disp, mode,padding_mode,align_corners)
    else:
        raise ValueError("ValueError: warping direction should be 'l2r' or 'r2l'")

def stereo_warp_r2l(right_img, left_disp, mode='bilinear',padding_mode="zeros",align_corners=False, grid_sample_type="pytorch", gt_map=None):
    return apply_disparity(right_img, -left_disp, mode,padding_mode,align_corners, grid_sample_type, gt_map) # from left view to right view projection, x_base-left_disparity

def stereo_warp_l2r(left_img, right_disp, mode='bilinear',padding_mode="zeros",align_corners=False, grid_sample_type="pytorch", gt_map=None):
    return apply_disparity(left_img,  right_disp, mode,padding_mode,align_corners, grid_sample_type, gt_map) 

def apply_disparity(img, disp, mode='bilinear',padding_mode="zeros",align_corners=False, grid_sample_type="pytorch", gt_map=None):
    batch_size, _, height, width = img.shape
    y_base, x_base = torch.meshgrid(
                            torch.linspace(0, height-1., height, dtype=disp.dtype, device=img.device),
                            torch.linspace(0,  width-1.,  width, dtype=disp.dtype, device=img.device))
    # Apply shift in X direction
    x_shifts = disp.squeeze(1) # Disparity is passed in NCHW format with 1 channel
    # norm for grid_sample 
    flow_field_x = x_base.unsqueeze(0)#.expand(x_shifts.shape)
    flow_field_x = (flow_field_x+x_shifts) / (width-1)
    flow_field_y = y_base / (height-1)
    flow_field_y = flow_field_y.unsqueeze(0).expand(x_shifts.shape)
    flow_field = torch.stack((flow_field_x, flow_field_y), dim=3) # N H W 2
    flow_field = (flow_field-0.5)*2.0
    # In grid_sample coordinates are assumed to be between -1 and 1
    # if grid_sample_type=="python_grad2":
    #     output = grid_sample_hessian(img, flow_field, )
    # if grid_sample_type=="pytorch_grad2":
    #     output = grid_sample_2d(img, flow_field, padding_mode=padding_mode,align_corners=align_corners)
    # if grid_sample_type=="esm":
    #     output = grid_sample_2d_esm(img, flow_field, gt_map, padding_mode=padding_mode,align_corners=align_corners)
    # if grid_sample_type=="fc":
    #     output = grid_sample_2d_fc(img, flow_field, gt_map, padding_mode=padding_mode,align_corners=align_corners)
    # if grid_sample_type=="ic":
    #     output = grid_sample_2d_ic(img, flow_field, gt_map, padding_mode=padding_mode,align_corners=align_corners)
    if grid_sample_type=="pytorch":
        output = F.grid_sample(img, flow_field, mode=mode,padding_mode=padding_mode,align_corners=align_corners)
    return output

def apply_disparity_v2(img, disp, mode='bilinear',padding_mode="zeros",align_corners=False):
    """
    Another implements, the warped result is a little different from the above one. about tensor(0.0026, device='cuda:0')
    speed is almost the same as the above one. 
    """
    batch_size, _, height, width = img.shape
    y_base, x_base = torch.meshgrid(
                            torch.linspace(-1.0 , 1.0 , height, dtype=disp.dtype, device=img.device),
                            torch.linspace(-1.0,  1.0,  width, dtype=disp.dtype, device=img.device))
    # Apply shift in X direction
    x_shifts = disp.squeeze(1) # Disparity is passed in NCHW format with 1 channel
    # norm for grid_sample 
    x_shifts = 2.0*x_shifts / (width-1)
    flow_field_x = x_base.unsqueeze(0) + x_shifts #.expand(x_shifts.shape)
    flow_field_y = y_base.unsqueeze(0).expand(x_shifts.shape)
    flow_field = torch.stack((flow_field_x, flow_field_y), dim=3) # N H W 2
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, flow_field, mode=mode,padding_mode=padding_mode,align_corners=align_corners)
    return output

# the old API 
def generate_image_left(img, disp,mod="dense"):
    return apply_disparity_old(img, -disp,mod, name="left_mask") # from left view to right view projection, x_base-left_disparity


def apply_disparity_old(img, disp,mod="dense",name=None):
    batch_size, _, height, width = img.size()
    y_base, x_base = torch.meshgrid(
                            torch.linspace(0.0 , height - 1.0 , height),
                            torch.linspace(0.0,   width - 1.0,  width))
    y_base, x_base =  y_base.unsqueeze(0).repeat(batch_size,1,1).to(img.device), \
                        x_base.unsqueeze(0).repeat(batch_size,1,1).to(img.device)
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # norm for grid_sample 
    flow_field[...,0] /=(width-1)
    flow_field[...,1] /=(height-1)
    flow_field = (flow_field-0.5)*2.0
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, flow_field, mode='bilinear',padding_mode='zeros',align_corners=False)
    return output


def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img
def load_depth_midair(z_path):
    depth = open_float16(z_path)
    w, h = depth.shape
    #print("depth shape >>.", depth.shape)
    f = w/2
    # tranfrom to camera coord
    x_i, y_i = np.meshgrid(
                                np.arange(0,w,1),
                                np.arange(0,h,1), indexing="xy")
    #print("xi >> \n", x_i)
    #print("y_i >> \n ", y_i)
    
    r_xy = depth / np.sqrt((x_i-(h/2))**2 + (y_i-(h/2))**2 + f**2)
    #x_c = r_xy*(x_i-h/2)
    #y_c = r_xy*(y_i-h/2)
    z_c = r_xy*(f)
    #z_c[depth>=65500.0] = 65500.0
    #print("z_c >> \n ", z_c)
    return z_c #np.expand_dims(z_c, axis=0)


if __name__ == '__main__':
    import time
    img = torch.rand(1,3,256,512).cuda()
    disp = torch.randint(0, 192, (1,1,256,512)).float().cuda()
    
    t0 = time.time()
    warped2 = generate_image_left(img, disp,mod="dense")
    t1 = time.time()
    print("old time: ", t1-t0)
    #print(warped2.shape)
    #print(warped2)

    t0 = time.time()
    warped2 = generate_image_left(img, disp,mod="dense")
    t1 = time.time()
    print("old time: ", t1-t0)
    #print(warped2.shape)
    #print(warped2)

    t0 = time.time()
    warped = stereo_warp_r2l(img, disp,)
    t1 = time.time()
    print("my time: ", t1-t0)
    #print(warped.shape)
    #print(warped)

    t0 = time.time()
    warped2 = generate_image_left(img, disp,mod="dense")
    t1 = time.time()
    print("old time: ", t1-t0)
    #print(warped2.shape)
    #print(warped2)

    #print(warped)
    #print(warped2)
    print(torch.sum(warped-warped2))
    assert warped.equal(warped2)


    import cv2
    from PIL import Image
    id = 100

    ref_img = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/image_0/000{:0>3}.JPEG".format(id)
    cur_img = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/image_1/000{:0>3}.JPEG".format(id)
    ref_depth = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/depth/000{:0>3}.PNG".format(id)
    cur_depth = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/depth/000{:0>3}.PNG".format(id+1)
    ref_img = cv2.imread(ref_img)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    #ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    h,w = ref_img.shape[:2] 
    ref_img = ref_img.reshape(1,h,w,3)
    print(ref_img.shape)
    
    ref_img = torch.FloatTensor(ref_img).permute(0,3,1,2)
    cur_img = cv2.imread(cur_img)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
    #cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2GRAY)
    cur_img = torch.FloatTensor(cur_img).reshape(1,h,w,3).permute(0,3,1,2)
    #ref_depth = 0.01 * cv2.imread(ref_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)   #load_depth_midair(ref_depth))
    ref_depth =  load_depth_midair(ref_depth) #cv2.imread(ref_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)   #load_depth_midair(ref_depth))
    ref_depth = torch.FloatTensor(ref_depth.reshape(1, 1, h, w))
    cur_depth = load_depth_midair(cur_depth) #cv2.imread(cur_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)   #load_depth_midair(ref_depth))
    cur_depth = torch.FloatTensor(cur_depth.reshape(1, 1, h, w))
    #vis_depth_tensor(ref_depth.unsqueeze(0), "/home/ziliu/vis/midair" "midair_ref_depth")
    #cv2.imwrite( "/home/ziliu/vis/midair/ref_img.png", ref_img.cpu().numpy()[0])
    #cv2.imwrite("/home/ziliu/vis/midair/cur_img.png",cur_img.cpu().numpy()[0])
    ref_disp = 1. * 512 / ref_depth

    K = torch.FloatTensor([512.0, 0.0, 512.0, 0.0, 0.0, 512.0, 512.0, 0.0, 0.0, 0.0, 1.0,\
                         0.0, 0.0, 0.0, 0.0, 1.0]).reshape(1, 4, 4)
    #K = torch.FloatTensor([7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 4.688783000000e+01, 
    #                    0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01,
    #                        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03]).reshape((3,4))
    #K = torch.cat([K, torch.FloatTensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)], dim=0)
    #imask = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/stereo_occlusion/000{:0>3}.PNG".format(id)
    #imask = cv2.imread(imask, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #imask[imask==255] = 1
    #imask = torch.FloatTensor(imask).reshape(h, w)
    imask = torch.ones((h, w), dtype=torch.float32,)
    imask = imask * (ref_depth > 1.0) * (ref_depth < 80.0)
    

    gt_posetxt = "/home/ziliu/mydata/MidAir_kittiformat/training/gt_pose_cam0/sunny_0000.txt"
    #gt_posetxt = "/home/ziliu/mydata/kitti_odometry/pose_GT/09.txt"
    with open(gt_posetxt) as f:
        lines = f.readlines()
        T_ref = torch.FloatTensor([float(x) for x in lines[id].split()]).reshape(3, 4)
        T_cur = torch.FloatTensor([float(x) for x in lines[id+1].split()]).reshape(3, 4)
    T_ref = torch.cat([T_ref, torch.FloatTensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)], 0)
    T_cur = torch.cat([T_cur, torch.FloatTensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)], 0)
    cTr = torch.linalg.solve(T_cur, T_ref).reshape(1,4,4) # torch.linalg.inv(T_cur) @ T_ref
    rTc = torch.linalg.solve(T_ref, T_cur).reshape(1,4,4) # torch.linalg.inv(T_ref) @ T_cur

    wl= stereo_warp_r2l(cur_img, ref_disp, mode="bilinear" )
    vis_img_tensor(wl, "/home/ziliu/vis/midair", "midair_wl", ifnorm=False)
    vis_img_tensor(ref_img, "/home/ziliu/vis/midair", "midair_l", ifnorm=False)
    vis_depth_tensor(ref_disp, "/home/ziliu/vis/midair", "midair_ref_depth")
    print(torch.sum(wl[wl!=0]-ref_img[wl!=0]))
    print(wl[wl!=0]-ref_img[wl!=0])
    print(torch.sum((wl[wl!=0]-ref_img[wl!=0])>10))
    vis_img_tensor(torch.abs(wl-ref_img), "/home/ziliu/vis/midair", "diff", ifnorm=False)

    """
    the new API achieves 30x-50x speed up.
    old time:  0.03488278388977051
    old time:  0.02023601531982422
    my time:  0.0006849765777587891
    old time:  0.03131461143493652
    """


    