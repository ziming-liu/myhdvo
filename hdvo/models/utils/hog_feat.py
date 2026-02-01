'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-29 14:30:02
LastEditors: Ziming Liu
LastEditTime: 2024-01-21 00:34:00
codebase: https://github.com/facebookresearch/SlowFast/blob/fcf407ec8252e66313f21aca782ff00b921e5071/slowfast/models/operators.py#L66
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hogBGR import HOGvisualized,HOG

def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()



class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3
        )
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3
        )
        # fix bug by ziming: to corresponding h, w, the order is gy, gx, instead of gx, gy as in the original code
        norm_rgb = torch.stack([gy_rgb, gx_rgb], dim=-1).norm(dim=-1) 
        phase = torch.atan2(gy_rgb, gx_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros(
            (b, c, self.nbins, h, w), dtype=torch.float, device=x.device
        )
        #print("out shape: ", out.shape)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(
                    h, self.gaussian_window
                )
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, w // self.gaussian_window])
                #print("temp_gkern shape", temp_gkern.shape)
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)
        #print("out2 shape", out.shape)


        out = out.unfold(3, self.pool, self.pool)
        #print("unfold3 shape", out.shape)
        out = out.unfold(4, self.pool, self.pool)
        #print("unfold4 shape", out.shape)
        out = out.sum(dim=[-1, -2])
        #print("sum shape", out.shape)

        

        out = torch.nn.functional.normalize(out, p=2, dim=2)
        
        
        """ 
        hog = out[0].clone().permute(2,3,0,1)
        hog = torch.flip(hog,[3])
        hog = hog.data.numpy()
        res = HOGvisualized.visualize_HOG(hog,8)
        print("res shape", res.shape)
        cv2.imshow("HOG ours",res)
        cv2.waitKey(0)
        
        tmp_hog = out[0].reshape(-1, out.shape[-2], out.shape[-1])
        tmp_hog = tmp_hog.permute(1,2,0)
        tmp_hog = (tmp_hog.sum(2)).numpy()
        vis = np.uint8(tmp_hog)*225
        cv2.imshow('HOG features', vis)
        cv2.waitKey(0)
        """
        return out

if __name__ == '__main__':
    import cv2
    import numpy as np

    image = cv2.imread('/Users/ziming/Downloads/kittistereo2012/data_stereo_flow/training/colored_0/000002_11.png')
    image = cv2.resize(image, (512, 224))
    cell_hogfeature = HOG.Cell_HOG(image,cell_size=(4,4))
    hog = HOG.BlockNorm_HOG(cell_hogfeature,block_size=(1,1))
    print("downsampled HOG shape: ", hog.shape)
    res = HOGvisualized.visualize_HOG(hog[:,:,:,0,0,:],cell_size=4)
    cv2.imshow("HOG feature with block normaliztion",res)
    cv2.imshow('rgb image', image)
    """ 
    from skimage.feature import hog
    from skimage import io
    im = io.imread('/Users/ziming/Downloads/kittistereo2012/data_stereo_flow/training/colored_0/000002_11.png', as_gray=True)
    normalised_blocks, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualize=True)
    print(hog_image.shape)
    print(normalised_blocks.shape)
    print(hog_image)
    cv2.imshow('hog', hog_image*100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #io.imshow(hog_image)

    exit()
    """
 
    

    hog_layer = HOGLayerC(nbins=9, pool=8, gaussian_window=0)

    img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    print(img.shape)
 
    #features = hog_layer(img)
    tmp_hog = hog_layer(img).flatten(1, 2)  # return B C H W
    print("tmp_hog shape: ", tmp_hog.shape)

    unfold_size = tmp_hog.shape[-1] // 8 #img.shape[-1]
    print("unfold_size: ", unfold_size)
    tmp_hog = tmp_hog.permute(0, 2, 3, 1)
    print("tmp_hog shape: ", tmp_hog.shape)
    tmp_hog = tmp_hog.unfold(1, unfold_size, unfold_size)
    print("tmp_hog shape: ", tmp_hog.shape)
    tmp_hog = tmp_hog.unfold(2, unfold_size, unfold_size)
    print("tmp_hog shape: ", tmp_hog.shape)
    tmp_hog = tmp_hog.flatten(1, 2)
    print("tmp_hog shape: ", tmp_hog.shape)
    tmp_hog = tmp_hog.flatten(2)
    print("tmp_hog shape: ", tmp_hog.shape)
    
    #tmp_hog = tmp_hog[output_mask]
    print("tmp_hog shape: ", tmp_hog.shape)
    #features = tmp_hog.sum(2, keepdim=True)/9
    #features = features[0,:,0,:,:].permute(1, 2, 0).numpy()
    tmp_hog = tmp_hog.permute(1,2,0).numpy()
    # 将特征图转换为可视化格式
    vis = np.uint8(tmp_hog)*225
    #vis = cv2.resize(vis, (image.shape[1], image.shape[0]))

    # 显示特征图
    #cv2.imshow('HOG features', vis)
    #cv2.waitKey(0)

    #cv2.destroyAllWindows()
