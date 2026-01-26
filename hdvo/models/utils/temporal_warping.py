import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from hdvo.core.visulization import vis_depth_tensor,vis_img_tensor
from .grid_sample_hessian import grid_sample_hessian
#try:
# from hdvo.models.utils.cuda_gridsample_grad2 import grid_sample_2d, grid_sample_2d_esm, grid_sample_2d_fc, grid_sample_2d_ic
#except:
#    print("Warning: cuda_gridsample_grad2 is not compiled, please compile it first. load gcc, cmake")

def temporal_warp_c2r(cimg, rdepth, cTr, K, invK, mode="bilinear",  padding_mode="zeros", align_corners=False, eps=1e-7, grid_sample_type="pytorch", gt_map=None):
    return temporal_warp_core(cimg, rdepth, cTr, K, invK, mode,  padding_mode, align_corners, eps, grid_sample_type, gt_map)

def temporal_warp_r2c(rimg, cdepth, rTc, K, invK, mode="bilinear",  padding_mode="zeros", align_corners=False, eps=1e-7, grid_sample_type="pytorch", gt_map=None):
    return temporal_warp_core(rimg, cdepth, rTc, K, invK, mode,  padding_mode, align_corners, eps, grid_sample_type, gt_map)

def temporal_warp_core(img, depth, T, K, invK, mode="bilinear",  padding_mode="zeros", align_corners=False, eps=1e-7, grid_sample_type="pytorch", gt_map=None):
    #assert K.shape[-1]==4, f"K shape should be B 4 4, but got {K.shape}"
    #assert invK.shape[-1]==4, f"invK shape should be B 4 4, but got {invK.shape}"
    #assert T.shape[-1]==4, f"T shape should be B 3 4, but got {T.shape}"
    #assert T.shape[-2]==4, f"T shape should be B 4 4, but got {T.shape}"
    
    batch_size, C, H, W = img.shape
    # init coordinates
    y_base, x_base = torch.meshgrid(
                            torch.linspace(0 , H-1, H, dtype=depth.dtype, device=img.device),
                            torch.linspace(0,  W-1,  W, dtype=depth.dtype, device=img.device))
    x_base = x_base.reshape(-1)
    y_base = y_base.reshape(-1)
    ones_ = torch.ones((H*W), dtype=depth.dtype, device=img.device)
    pix_coords = torch.stack((x_base, y_base, ones_), dim=0).unsqueeze(0).expand(batch_size, 3, H*W)
     # B 3 H*W
    
    assert batch_size==invK.shape[0], f"keep batch size {batch_size} == {invK.shape[0]}"
    
    cam_points = torch.bmm(invK[:, :3, :3], pix_coords) # B 3 H*W # the last row is 1111
    #cam_points = torch.linalg.solve(K[:, :3, :3], pix_coords) # B 3 H*W # the last row is 1111
    cam_points = depth.reshape(batch_size, 1, H*W) * cam_points # B 3 H*W
    cam_points = torch.cat([cam_points,\
                 torch.ones((batch_size, 1, H*W), dtype=depth.dtype, device=img.device)], 1) # B 4 H*W
    
    P = torch.bmm(K[:,:3,:3], T[:,:3,:4]) # B 3 4
    cam_points = torch.bmm(P, cam_points) # B 3 H*W
    
    pix_coords = cam_points[:, :2, ] / (cam_points[:, 2:3, ] + eps)
    pix_coords = pix_coords.permute(0, 2, 1).reshape(batch_size, H, W, 2) # B H, W 2

    # warp image
    pix_coords[..., 0] = pix_coords[..., 0] / (W - 1)
    pix_coords[..., 1] = pix_coords[..., 1] / (H - 1)
    pix_coords = (pix_coords - 0.5) * 2

    # if grid_sample_type=="python_grad2":
    #     output =  grid_sample_hessian(img, pix_coords, )
    # if grid_sample_type=="pytorch_grad2":
    #     output =  grid_sample_2d(img, pix_coords, padding_mode=padding_mode, \
    #                                    align_corners=align_corners)
    # if grid_sample_type=="esm":
    #     output = grid_sample_2d_esm(img, pix_coords, gt_map, padding_mode=padding_mode, \
    #                                    align_corners=align_corners)
    # if grid_sample_type=="fc":
    #     output = grid_sample_2d_fc(img, pix_coords, gt_map, padding_mode=padding_mode, \
    #                                    align_corners=align_corners)
    # if grid_sample_type=="ic":
    #     output = grid_sample_2d_ic(img, pix_coords, gt_map, padding_mode=padding_mode, \
    #                                    align_corners=align_corners)
    if grid_sample_type=="pytorch":
        output = F.grid_sample(img, pix_coords, mode=mode, padding_mode=padding_mode, \
                                       align_corners=align_corners)
    return output



def temporal_warp_old(source_map, target_depth, T, K, invK, stereo_view="left", padding_mode="zeros", eps=1e-7):
    '''
    description: perform backward warping operation. the projection and the sampling operation is opposite for backward warping. 
    parameter: {*}
        source_map, to be sampled to genrate the target warped map
        target_depth, used for compute projection relations
        T, generate projection relations
        stereo_view, "left" or "right" view
        padding_mode=self.padding_mode 
    return: {*}
        warped_target, 
    '''        
    B, C, H, W = target_depth.shape
    # init coordinates
    meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.FloatTensor(id_coords)
    id_coords_ = nn.Parameter(id_coords, requires_grad=False)
    ones_ = nn.Parameter(torch.ones(1, 1, H * W),requires_grad=False).cuda() # assume bs ==1
    pix_coords = torch.unsqueeze(torch.stack(
        [id_coords_[0].view(-1), id_coords_[1].view(-1)], 0), 0).cuda()
    pix_coords = pix_coords.repeat(1, 1, 1) # assum bs ==1
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones_], 1),requires_grad=False)
    # back project 
    batch_size = target_depth.shape[0] # because the bs of the last iteration maybe smaller than setted bs, make it dynamic
    assert batch_size==invK.shape[0], f"keep batch size {batch_size} == {invK.shape[0]}"
    pix_coords = pix_coords.repeat(batch_size,1,1)
    ones = ones_.repeat(batch_size,1,1)

    cam_points = torch.bmm(invK[:, :3, :3], pix_coords.type_as(invK))
    cam_points = target_depth.view(batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)
    # project 3D 
    if K.shape[-1]==3:
        P = torch.bmm(K, T[:,:3,:])[:, :3, :]
    elif K.shape[-1]==4:
        P = torch.bmm(K, T)[:, :3, :]
    cam_points = torch.bmm(P, cam_points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
    pix_coords = pix_coords.view(batch_size, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2
    #cam_points = self.layer_backproject_depth(target_depth,  invK)
    #pix_coordi = self.layer_project(cam_points, K, T)
    warped_target_map = F.grid_sample(source_map, pix_coords, padding_mode=padding_mode, align_corners=False)
    return warped_target_map.type_as(source_map)


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
    cimg = torch.rand(1, 3, 256, 512).cuda()
    depth = torch.randint(1, 80, (1, 1, 256, 512)).float().cuda()
    K = torch.eye(3).unsqueeze(0).cuda()
    T = torch.eye(4).unsqueeze(0).cuda()
    invK = torch.eye(3).unsqueeze(0).cuda()
    t0 = time.time()
    
    warped_new = temporal_warp_r2c(cimg, depth, T, K, invK, "zeros")
    t1 = time.time()
    print("new api time: ", t1-t0)
    t0 = time.time()
    warped_old = temporal_warp_old(cimg, depth, T, K, invK, "zeros")
    t1 = time.time()
    print("old api time: ", t1-t0)

    t0 = time.time()
    warped_new = temporal_warp_r2c(cimg, depth, T, K, invK, "zeros")
    t1 = time.time()
    print("new api time: ", t1-t0)

    t0 = time.time()
    warped_old = temporal_warp_old(cimg, depth, T, K, invK, "zeros")
    t1 = time.time()
    print("old api time: ", t1-t0)
    print("warped new ", warped_new)
    print("warped old ", warped_old)
    print(warped_new.equal(warped_old))
    
    import cv2
    from PIL import Image
    id = 0

    ref_img = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/image_0/000{:0>3}.JPEG".format(id)
    cur_img = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/image_0/000{:0>3}.JPEG".format(id+1)
    ref_depth = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/depth/000{:0>3}.PNG".format(id)
    cur_depth = "/home/ziliu/mydata/MidAir_kittiformat/training/sequences/sunny_0000/depth/000{:0>3}.PNG".format(id+1)
    ref_img = cv2.imread(ref_img)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    h,w = ref_img.shape[:2] 
    ref_img = ref_img.reshape(1,1,h,w)
    print(ref_img.shape)
    
    ref_img = torch.FloatTensor(ref_img)#.unsqueeze(0)
    cur_img = cv2.imread(cur_img)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2GRAY)
    cur_img = torch.FloatTensor(cur_img).reshape(1,1,h,w)#.unsqueeze(0)
    #ref_depth = 0.01 * cv2.imread(ref_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)   #load_depth_midair(ref_depth))
    ref_depth =  load_depth_midair(ref_depth) #cv2.imread(ref_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)   #load_depth_midair(ref_depth))
    ref_depth = torch.FloatTensor(ref_depth.reshape(1, 1, h, w))
    cur_depth = load_depth_midair(cur_depth) #cv2.imread(cur_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)   #load_depth_midair(ref_depth))
    cur_depth = torch.FloatTensor(cur_depth.reshape(1, 1, h, w))
    #vis_depth_tensor(ref_depth.unsqueeze(0), "/home/ziliu/vis/midair" "midair_ref_depth")
    #cv2.imwrite( "/home/ziliu/vis/midair/ref_img.png", ref_img.cpu().numpy()[0])
    #cv2.imwrite("/home/ziliu/vis/midair/cur_img.png",cur_img.cpu().numpy()[0])

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

    wr = temporal_warp_c2r(cur_img, ref_depth, cTr, K, torch.linalg.inv(K), )
    vis_depth_tensor(wr, "/home/ziliu/vis/midair", "midair_wr")
    vis_depth_tensor(ref_img,"/home/ziliu/vis/midair", "midair_ref")

    wc = temporal_warp_r2c(ref_img, cur_depth, rTc, K, torch.linalg.inv(K), )
    vis_depth_tensor(wc, "/home/ziliu/vis/midair", "midair_wc")
    vis_depth_tensor(cur_img,"/home/ziliu/vis/midair", "midair_cur")
    
    """
    To modify the code, we achieves 50x-100x speed up.
        new api time:  0.63808274269104
        old api time:  0.04565930366516113
        new api time:  0.00045871734619140625
        old api time:  0.038934946060180664
    """