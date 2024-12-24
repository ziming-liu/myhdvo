'''
Author: Ziming Liu
Date: 2022-07-01 00:39:10
LastEditors: Ziming Liu
LastEditTime: 2024-01-29 18:56:09
Description: ...
Dependent packages: don't need any extral dependency
'''
from abc import ABCMeta, abstractmethod
import warnings

import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from ...core import tensor_img_denorm

from hdvo.models.losses import ZNCC, ZNCCLoss

from ..builder import build_loss

from ..registry import HEADS,MASKS

from hdvo.core.visulization import *
import os

try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except:
    warnings("Not intall mobile_sam")
try:
    from ultralytics import SAM
except:
    warnings("Not intall ultralytics")

try:
    import sys
    sys.path.append("/home/ziliu/mydata/segment-anything")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except:
    warnings("Not intall tinysam")


import random
from ...core import tensor_img_denorm


@MASKS.register_module()
class HomoMaskTinySAM(nn.Module):
    def __init__(self, num_views=2, num_frames=2,  num_level=3,loss_func=["spr","tpr"], 
                 default_view="left", kernel1_size=(5,5),  kernel2_size=(5,5), 
                 mask_percent=None, threshold=0.03, sam_threshold=0.8, n_samples=10, old_api=False):
        super().__init__()
        self.num_views = num_views
        self.num_frames = num_frames
        self.num_level= num_level
        self.default_view = default_view
        self.mask_percent = mask_percent
        self.loss_func = loss_func
        self.old_api = old_api
        if self.num_views == 1:
            self.views = [default_view]
        else:
            self.views = ["left", "right"]
        #self.input = [[None for j in range(num_views)] for i in range(num_frames)]
        #self.output = [[None for j in range(num_views)] for i in range(num_frames)]
        self.kernel1_size = kernel1_size
        self.kernel2_size = kernel2_size
        self.threshold = threshold
        self.sam_threshold = sam_threshold
        self.meanpool2d = torch.nn.AvgPool2d(kernel_size=self.kernel1_size, stride=(1,1), padding= (int((self.kernel1_size[0]-1)/2), int((self.kernel1_size[1]-1)/2))).cuda()
        self.pool2d = torch.nn.MaxPool2d(kernel_size=self.kernel2_size, stride=(1,1), padding= (int((self.kernel2_size[0]-1)/2), int((self.kernel2_size[1]-1)/2))).cuda()

        
        #model_type = "vit_t"
        #self.sam = sam_model_registry[model_type](checkpoint="/home/ziliu/mydata/TinySAM/weights/tinysam.pth")
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.sam.to(device=device)
        #self.predictor = SamPredictor(self.sam)
        model_type = "vit_l"
        sam_checkpoint = "/home/ziliu/mydata/segment-anything/sam_vit_l_0b3195.pth"#sam_vit_h_4b8939.pth"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        mobile_sam.mask_threshold = 0
        print("threshold", mobile_sam.mask_threshold)
        self.predictor = SamPredictor(mobile_sam)

        self.n_samples = n_samples # random sample n points for SAM segmentation
        

    def _each_homo_mask(self, x):
        b, c, h, w = x.shape
        x_mean = self.meanpool2d(x)
        x_mean = self.meanpool2d(x_mean)
        x= torch.abs(x-x_mean)
        x = self.pool2d(x)

        mask = torch.ones(x.shape, device=x.device)
        mask[x<(self.threshold)] = 0 # mask homogenous region
        mask_out, max_idx = torch.max(mask,1,True) # return bx1xhxw
        return mask_out, -x


    def forward(self, x, frame_names=None):
        '''
        description:  
        return: {*} [[bx1xhxw], ...]  homogeneous masks
        '''        
        #assert x.shape[1] == 1, "use gray scale image"
        lam_mask, err =  self._each_homo_mask(x)#.detach()
        lam_mask = err.sum(1,True)
        print("lam mask", lam_mask.shape)
        b,c,h,w = lam_mask.shape
        
        # random sample 10 points from lam mask
        final_masks =[]
        #x = tensor_img_denorm(x, mean=[88.78708011161852, 93.43778497818349, 91.33551888646076], std=[80.93941240862273, 81.55742718042109, 82.55097977909143])
        for i in range(x.shape[0]):
            #image = x[i] # 3 h w
            image = cv2.imread(frame_names[i])
            image = cv2.resize(image, (1024,1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
            #mask = (lam_mask[i]==0).float()
            mask = lam_mask[i]
            mask = F.interpolate(mask.unsqueeze(0), size=(1024,1024), mode="bilinear").squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), scale_factor=1/4, mode="bilinear").squeeze(0)
            #vis_depth_tensor(mask, "/home/ziliu/vis/tinysam_homo", "lam_mask")
            
            # Pad
            #mh, mw = mask.shape[-2:]
            #padh = 256 - mh
            #padw = 256 - mw
            #mask = F.pad(mask, (0, padw, 0, padh))
            
            input_point = np.array([[400, 400]])
            input_label = np.array([1])
            #mask[mask==1] = 10
            coords = (mask.squeeze()>0).nonzero()
            N = coords.shape[0]
            #index = torch.LongTensor(random.sample(range(N), int(N/5))).cuda()
            #print(index)
            #coords = torch.index_select(coords, 0, index)
            labels = torch.ones([coords.shape[0]])
            print(coords.shape)
            print(labels.shape)
            print(mask)
            masks, ious, low_res_masks_np = self.predictor.predict(
                point_coords=None, #coords.cpu().numpy(),
                point_labels=None, #labels.cpu().numpy(),
                box=None,
                mask_input = mask.cpu().numpy(),
                multimask_output=True,
                )
            #print(low_res_masks_np)
            #print(masks.shape)
            masks = F.interpolate(torch.FloatTensor(masks).unsqueeze(0), size=(h,w), mode="nearest").squeeze(0)
            for i in range(masks.shape[0]):
                vis_depth_tensor(torch.FloatTensor(masks[i]), "/home/ziliu/vis/tinysam_homo", f"mask{i}")
            new_seg = np.array(masks[np.argmax(ious)],order="F", dtype="uint8")
            vis_depth_tensor(torch.FloatTensor(new_seg), "/home/ziliu/vis/tinysam_homo", f"finalmask")
            final_masks.append(new_seg)
            """ 
            mask = lam_mask[i].squeeze()
            coords = (mask==0).nonzero() # mask==0 coordinates # https://blog.csdn.net/judgechen1997/article/details/105820709
            # random n positions
            N = coords.shape[0]
            index = torch.LongTensor(random.sample(range(N), self.n_samples)).cuda()
            #print(index)
            rand_coords = torch.index_select(coords, 0, index)
            labels = torch.ones([rand_coords.shape[0]]).cuda()

            print("rand coords", rand_coords.shape)
            #input_point = torch.LongTensor([[200, 400]]).cuda().unsqueeze(0)
            #input_label = torch.LongTensor([1]).cuda().unsqueeze(0)
            item = dict(image=image, original_size=(h,w),\
                         point_coords=rand_coords.unsqueeze(0), point_labels= labels.unsqueeze(0))
            """
            """ 
            #print("mask ", mask.unsqueeze(0).shape)
            mask_inv = (mask==0).float()
            #print("image shape",image.shape )
            # Pad
            padh = self.sam.image_encoder.img_size - h
            padw = self.sam.image_encoder.img_size - w
            mask_inv = F.pad(mask_inv, (0, padw, 0, padh))
            mask_inv = F.interpolate(mask_inv.unsqueeze(0), scale_factor=1/4, mode='nearest',).squeeze(0)
            #print("mask_inv shape", mask_inv.shape)
            item = dict(image=image, original_size=(h,w), )
                         #mask_inputs=mask_inv)
            input.append(item)
            """
        """" 
        outs = self.sam(input, True)
        masks = torch.cat([outs[i]["masks"] for i in range(b)], 0 )# B C H W 
        masks = (masks==0).float()
        print("outputmask of sam ", masks.shape)
        #total_masks = torch.cat([lam_mask, masks], 1) # b c+1 h w 
        total_masks = masks
        total_masks, idx = total_masks.min(1,True)
        """
        final_masks = torch.from_numpy(np.array(final_masks)).cuda()
        

        return final_masks

 
        

        
        