'''
Author: Ziming Liu
Date: 2022-06-22 15:55:19
LastEditors: Ziming Liu
LastEditTime: 2024-01-22 23:50:55
Description: ...
Dependent packages: don't need any extral dependency
'''
import torch
import mmcv
import numpy as np

def tensor_img_denorm(img, mean=[88.78708011161852, 93.43778497818349, 91.33551888646076], std=[80.93941240862273, 81.55742718042109, 82.55097977909143]):
    '''
    description: de-normalize pytorch tensor img shaping like bxcxhxw.  
    return: {*}
    '''    
    device = img.device
    if len(img.shape)==3:
        img = img.unsqueeze(0)
    assert len(img.shape)==4
    img  = img.detach().cpu().numpy().transpose(0,2,3,1)
    b = img.shape[0]
    denorm_imgs = []
    for bi in range(b):
        denorm_imgs.append(mmcv.image.imdenormalize(img[bi], mean=np.array(mean), std=np.array(std), ))#to_rgb=False))
    denorm_img = torch.FloatTensor(np.stack(denorm_imgs)).permute(0,3,1,2)
    return denorm_img.to(device)