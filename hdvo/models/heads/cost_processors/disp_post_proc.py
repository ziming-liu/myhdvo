'''
Author: Ziming Liu
Date: 2022-11-03 00:33:15
LastEditors: Ziming Liu
LastEditTime: 2023-03-23 10:49:15
Description: ...
Dependent packages: kornia, torch
'''
import torch
import numpy as np
from torch.nn import functional as F
#import kornia
from hdvo.core.visulization import vis_img_tensor,vis_depth_tensor


def apply_disparity( img, disp,mod="dense",name=None, padding_mode="zeros", align_corners=False):
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
    output = F.grid_sample(img, flow_field, mode='bilinear',padding_mode=padding_mode,align_corners=align_corners)

    return output#,visible_mask

def generate_image_left(  img, disp,mod="dense"):
    return apply_disparity(img, -disp,mod, name="left_mask") # from left view to right view projection, x_base-left_disparity

def generate_image_right( img, disp,mod="dense"):
    return apply_disparity(img,  disp,mod, name="right_mask") 

def left_right_consistency(left_disp, right_disp, threshold=10):
    gene_left_disp = generate_image_left(right_disp, left_disp)
    gene_right_disp = generate_image_right(left_disp, right_disp)
    vis_depth_tensor(gene_left_disp,"/home/ziliu/vis/disp_post_porc","gene_left_disp")
    vis_depth_tensor(left_disp,"/home/ziliu/vis/disp_post_porc","left_disp")
    vis_depth_tensor(gene_right_disp,"/home/ziliu/vis/disp_post_porc","gene_right_disp")
    vis_depth_tensor(right_disp,"/home/ziliu/vis/disp_post_porc","right_disp")
    left_mask = (torch.abs(left_disp-gene_left_disp)>threshold)
    right_mask = (torch.abs(right_disp-gene_right_disp)>threshold)

    masked_left_disp = left_disp.clone()
    masked_left_disp[left_mask] = -1
    masked_right_disp = right_disp.clone()
    masked_right_disp[right_mask] = -1 

    return (masked_left_disp, masked_right_disp, left_mask, right_mask)

def fill_occlusion(masked_disp, ):
    B, _, H, W = masked_disp.shape
    new_disp = []
    for b in range(B):
        each_masked_disp = masked_disp[b,0]
        index_ = torch.nonzero(each_masked_disp==-1) 
        # only operate on -1 values' positions
        print("num to be filled >> ", {len(index_)})
        for i in range(len(index_)):
            h,w = index_[i]
            # find pL , left direction
            pl = w-1
            value_pl = torch.FloatTensor([999]).to(each_masked_disp.device)
            for pl in range(w-1,-1, -1):
                if each_masked_disp[h][pl]!=-1:
                    value_pl = each_masked_disp[h][pl]
                    break
            pr = w+1
            value_pr = torch.FloatTensor([999]).to(each_masked_disp.device)
            for pr in range(w+1,W,1):
                if each_masked_disp[h][pr]!=-1:
                    value_pr = each_masked_disp[h][pr]
                    break
            
            # fill occlu
            each_masked_disp[h][w] = torch.min(value_pl, value_pr) # keep value
        new_disp.append(each_masked_disp)
    return torch.stack(new_disp,0).unsqueeze(1) # B 1 H W 

#def median_filter(filled_disp, kernel_size=(3,3)):
#    assert len(filled_disp.shape)==4, f"input shape {filled_disp.shape} should be H C H W"
#    filtered_disp = kornia.filters.median_blur(filled_disp, kernel_size)
#    return filtered_disp

def disp_post_proc(left_disp, right_disp, threshold=10):
    print("LR consistency>>")
    masked_left_disp, masked_right_disp, left_mask, right_mask = \
         left_right_consistency(left_disp, right_disp, threshold=threshold)
    vis_depth_tensor(masked_left_disp,"/home/ziliu/vis/disp_post_porc","masked_left_disp")
    vis_depth_tensor(masked_right_disp,"/home/ziliu/vis/disp_post_porc","masked_right_disp")
    vis_depth_tensor(left_mask,"/home/ziliu/vis/disp_post_porc","left_mask")
    vis_depth_tensor(right_mask,"/home/ziliu/vis/disp_post_porc","right_mask")
    print("fill disp>>")
    filled_left_disp = fill_occlusion(masked_left_disp)
    filled_right_disp = fill_occlusion(masked_right_disp)
    vis_depth_tensor(filled_left_disp,"/home/ziliu/vis/disp_post_porc","filled_left_disp")
    vis_depth_tensor(filled_right_disp,"/home/ziliu/vis/disp_post_porc","filled_right_disp")
    print("filter median>>")
    #filtered_left_disp = median_filter(filled_left_disp)
    #filtered_right_disp = median_filter(filled_right_disp)
    #vis_depth_tensor(filtered_left_disp,"/home/ziliu/vis/disp_post_porc","filtered_left_disp")
    #vis_depth_tensor(filtered_right_disp,"/home/ziliu/vis/disp_post_porc","filtered_right_disp")

    #return (filtered_left_disp, filtered_right_disp)