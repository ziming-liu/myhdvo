'''
Author: Ziming Liu
Date: 2022-07-01 00:39:10
LastEditors: Ziming Liu
LastEditTime: 2023-11-17 17:01:14
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

from ..registry import HEADS

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

@HEADS.register_module()
class HomoMaskSAM(nn.Module):
    def __init__(self, num_views=2, num_frames=2,  num_level=3,loss_func=["spr","tpr"], 
                 default_view="left", kernel1_size=(5,5),  kernel2_size=(5,5), 
                 mask_percent=None, threshold=0.25, sam_threshold=0.8, old_api=False):
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

        #self.sam_model = SAM('mobile_sam.pt')
        self.sam_model = SAM('sam_l.pt')
        self.sam_model.info()  # display model information

        #model_type = "vit_t"
        #sam_checkpoint = "/home/ziliu/mydata/MobileSAM/weights/mobile_sam.pt"
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        #mobile_sam.to(device=device)
        #mobile_sam.eval()
        #self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        #self.predictor = SamPredictor(mobile_sam)

        

    def _each_homo_mask(self, x):
        b, c, h, w = x.shape
        x_mean = self.meanpool2d(x)
        x_mean = self.meanpool2d(x_mean)
        x= torch.abs(x-x_mean)
        x = self.pool2d(x)

        mask = torch.ones(x.shape, device=x.device)
        mask[x<(self.threshold)] = 0 # mask homogenous region
        mask_out, max_idx = torch.max(mask,1,True) # return bx1xhxw
        return mask_out


    def forward(self, x, frame_names):
        '''
        description:  
        return: {*} [[bx1xhxw], ...]  homogeneous masks
        '''        
        assert x.shape[1] == 1, "use gray scale image"
        lam_mask =  self._each_homo_mask(x).detach()
        b,c,h,w = lam_mask.shape
        batch_sam_masks = []
        for idx, frame in enumerate(frame_names):
        #for i in range(x.shape[0]):
            results = self.sam_model.predict(frame)[0]
            sam_masks = results.masks.data
            # N H W reshape to lam mask shape
            sam_masks = F.interpolate(sam_masks.unsqueeze(0).float(), size=(h,w),mode="nearest",).squeeze(0)
            """ 
            input = cv2.imread(frame)
            input = cv2.resize(input, (w,h), interpolation=cv2.INTER_LINEAR)
            #t0 = time.time()
            #sam_masks = self.mask_generator.generate(input)
            self.predictor.set_image(input)
            point_coords = torch.where(torch.logical_not(lam_mask[idx].squeeze())>0)
            point_coords = np.stack([point_coords[i].cpu().numpy() for i in range(len(point_coords))]).transpose()
            point_coords = point_coords[0:point_coords.shape[0]:point_coords.shape[0]//300,:]
            print(point_coords.shape)
            point_labels = np.ones(point_coords.shape[0])
            print(point_labels.shape)
            sam_masks, _, _ = self.predictor.predict(point_coords=point_coords,point_labels=point_labels)
            sam_masks = torch.FloatTensor(sam_masks).cuda()
            #torch.cuda.synchronize()
            #t1 = time.time()
            #print("time: ", t1-t0)
            #sam_masks = torch.stack([torch.FloatTensor(sam_masks[i]["segmentation"]).cuda() for i in range(len(sam_masks))])
            
            """
            
            batch_sam_masks.append(sam_masks)
        batch_sam_masks = torch.stack(batch_sam_masks).to(x.device) # B N H W 
        batch_refined_lam_mask = []
        for batch_idx in range(b):
            reference = lam_mask[batch_idx,0,...].bool()
            target_sams = batch_sam_masks[batch_idx,...]
            N, H, W  = target_sams.shape
            refined_lam_mask = torch.zeros((H,W),device=reference.device)
            #vis_depth_tensor(reference, "/home/ziliu/vis/semantic_lam", "lam")
            
            for sam_mask_idx in range(N):
                target_sam = target_sams[sam_mask_idx]
                #vis_depth_tensor(target_sam, "/home/ziliu/vis/semantic_lam", "target_sam")
                intersection = torch.logical_and(torch.logical_not(reference), target_sam)
                p = torch.sum(intersection.long()) / torch.sum(target_sam.long())
                if p>=self.sam_threshold:
                    refined_lam_mask = torch.logical_or(refined_lam_mask, target_sam)
            refined_lam_mask = torch.logical_not(refined_lam_mask)
            #vis_depth_tensor(refined_lam_mask, "/home/ziliu/vis/semantic_lam", "semantic_lam")
            batch_refined_lam_mask.append(refined_lam_mask)
        batch_refined_lam_mask = torch.stack(batch_refined_lam_mask).unsqueeze(1) # B 1 H W
        return batch_refined_lam_mask

 
        

        
        